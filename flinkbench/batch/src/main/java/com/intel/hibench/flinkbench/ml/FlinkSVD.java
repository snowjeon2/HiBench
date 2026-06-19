/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.hibench.flinkbench.ml;

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.util.Collector;

import java.util.List;
import java.util.Random;

/**
 * Flink batch truncated SVD via distributed Gram matrix (A^T A) computation +
 * driver-side power iteration for top-k singular values.
 * Uses ONE Flink job to avoid deep DAG chains from per-component IterativeDataSets.
 * Args: &lt;input_path&gt; &lt;num_examples&gt; &lt;num_features&gt; &lt;num_singular_values&gt;
 */
public class FlinkSVD {

    private static final int POWER_ITER  = 30;
    private static final int MAX_FEATURES = 500; // cap for Gram matrix size
    private static final int MAX_K        = 20;  // cap singular values to avoid slow driver loop

    public static void main(String[] args) throws Exception {
        if (args.length < 5) {
            System.err.println(
                "Usage: FlinkSVD <input_path> <num_examples> <num_features> <num_singular_values> <output_path>");
            System.exit(1);
        }
        final long numExamples     = Long.parseLong(args[1]);
        final int numFeatures      = Math.min(Integer.parseInt(args[2]), MAX_FEATURES);
        final int numSingularValues = Math.min(Integer.parseInt(args[3]), Math.min(MAX_K, numFeatures));
        final String outputPath    = args[4];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Generate random matrix rows: A ∈ R^{n × d}
        DataSet<double[]> matrix = env.generateSequence(0, numExamples - 1)
                .mapPartition(new MapPartitionFunction<Long, double[]>() {
                    @Override
                    public void mapPartition(Iterable<Long> values, Collector<double[]> out) {
                        for (Long seed : values) {
                            Random r = new Random(seed);
                            double[] row = new double[numFeatures];
                            for (int i = 0; i < numFeatures; i++) row[i] = r.nextGaussian();
                            out.collect(row);
                        }
                    }
                });

        // --- Single Flink job: compute Gram matrix A^T A (upper triangle) ---
        // Accumulator: upper triangle of d×d matrix, size = d*(d+1)/2
        final int d = numFeatures;
        final int triSize = d * (d + 1) / 2;

        DataSet<double[]> gramAcc = matrix
                .map(new MapFunction<double[], double[]>() {
                    @Override
                    public double[] map(double[] row) {
                        double[] acc = new double[triSize];
                        int idx = 0;
                        for (int i = 0; i < d; i++)
                            for (int j = i; j < d; j++)
                                acc[idx++] = row[i] * row[j];
                        return acc;
                    }
                })
                .reduce(new ReduceFunction<double[]>() {
                    @Override
                    public double[] reduce(double[] a, double[] b) {
                        double[] r = new double[triSize];
                        for (int i = 0; i < triSize; i++) r[i] = a[i] + b[i];
                        return r;
                    }
                });

        List<double[]> gramList = FlinkJobUtils.collect(gramAcc, env, "SVD");
        double[] gramFlat = gramList.get(0);

        // Reconstruct symmetric Gram matrix
        double[][] G = new double[d][d];
        {
            int idx = 0;
            for (int i = 0; i < d; i++) {
                for (int j = i; j < d; j++) {
                    G[i][j] = gramFlat[idx];
                    G[j][i] = gramFlat[idx];
                    idx++;
                }
            }
        }

        // --- Driver-side power iteration for top-k eigenvalues of A^T A ---
        // Singular values = sqrt(eigenvalues of A^T A)
        double[] singularValues = new double[numSingularValues];

        for (int s = 0; s < numSingularValues; s++) {
            double[] v = new double[d];
            {
                Random rng = new Random(s * 7777L);
                double norm = 0;
                for (int i = 0; i < d; i++) { v[i] = rng.nextGaussian(); norm += v[i]*v[i]; }
                norm = Math.sqrt(norm);
                for (int i = 0; i < d; i++) v[i] /= norm;
            }

            for (int iter = 0; iter < POWER_ITER; iter++) {
                double[] Gv = new double[d];
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++) Gv[i] += G[i][j] * v[j];
                double norm = 0;
                for (double x : Gv) norm += x * x;
                norm = Math.sqrt(Math.max(norm, 1e-12));
                for (int i = 0; i < d; i++) v[i] = Gv[i] / norm;
            }

            // Eigenvalue lambda = v^T G v
            double lambda = 0;
            double[] Gv = new double[d];
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++) Gv[i] += G[i][j] * v[j];
            for (int i = 0; i < d; i++) lambda += v[i] * Gv[i];

            singularValues[s] = Math.sqrt(Math.max(lambda, 0.0));

            // Deflate: G = G - lambda * v * v^T
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++) G[i][j] -= lambda * v[i] * v[j];
        }

        System.out.println("SVD top-" + numSingularValues + " singular values"
                + " (n=" + numExamples + ", d=" + d + "):");
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < numSingularValues; i++) {
            System.out.printf("  sigma[%d] = %.4f%n", i, singularValues[i]);
            sb.append("sigma[").append(i).append("]=").append(String.format("%.4f", singularValues[i])).append("\n");
        }
        writeResult(outputPath, sb.toString());
    }

    private static void writeResult(String hdfsPath, String content) throws java.io.IOException {
        org.apache.hadoop.conf.Configuration conf = new org.apache.hadoop.conf.Configuration();
        org.apache.hadoop.fs.Path dir = new org.apache.hadoop.fs.Path(hdfsPath);
        org.apache.hadoop.fs.FileSystem fs = dir.getFileSystem(conf);
        fs.mkdirs(dir);
        org.apache.hadoop.fs.Path file = new org.apache.hadoop.fs.Path(hdfsPath + "/part-00000");
        try (java.io.PrintStream out = new java.io.PrintStream(fs.create(file, true))) {
            out.print(content);
        }
    }
}
