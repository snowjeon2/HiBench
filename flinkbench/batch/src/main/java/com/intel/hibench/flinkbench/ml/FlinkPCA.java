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
 * Flink batch PCA via distributed covariance matrix computation + driver-side power iteration.
 * Uses ONE Flink job (covariance) to avoid deep DAG chains from per-component IterativeDataSets.
 * Args: &lt;input_path&gt; &lt;num_examples&gt; &lt;num_features&gt; &lt;k&gt;
 */
public class FlinkPCA {

    private static final int POWER_ITER = 30;

    public static void main(String[] args) throws Exception {
        if (args.length < 5) {
            System.err.println("Usage: FlinkPCA <input_path> <num_examples> <num_features> <k> <output_path>");
            System.exit(1);
        }
        final long numExamples = Long.parseLong(args[1]);
        final int numFeatures  = Integer.parseInt(args[2]);
        final int k            = Integer.parseInt(args[3]);
        final String outputPath = args[4];
        // Cap features to avoid huge covariance matrix (d*(d+1)/2 entries)
        final int d = Math.min(numFeatures, 500);

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Generate random matrix rows
        DataSet<double[]> matrix = env.generateSequence(0, numExamples - 1)
                .mapPartition(new MapPartitionFunction<Long, double[]>() {
                    @Override
                    public void mapPartition(Iterable<Long> values, Collector<double[]> out) {
                        for (Long seed : values) {
                            Random r = new Random(seed);
                            double[] row = new double[d];
                            for (int i = 0; i < d; i++) row[i] = r.nextGaussian();
                            out.collect(row);
                        }
                    }
                });

        // --- Single Flink job: compute packed [sum_i, sumsq_ij_upper_triangle] ---
        // Accumulator layout: [sum_0..sum_{d-1}, cov_upper_tri(d*(d+1)/2), count]
        final int triSize = d * (d + 1) / 2;
        final int accLen  = d + triSize + 1;

        DataSet<double[]> statsAcc = matrix
                .map(new MapFunction<double[], double[]>() {
                    @Override
                    public double[] map(double[] row) {
                        double[] acc = new double[accLen];
                        // sums
                        for (int i = 0; i < d; i++) acc[i] = row[i];
                        // upper triangle of outer product
                        int idx = d;
                        for (int i = 0; i < d; i++) {
                            for (int j = i; j < d; j++) {
                                acc[idx++] = row[i] * row[j];
                            }
                        }
                        acc[accLen - 1] = 1.0;
                        return acc;
                    }
                })
                .reduce(new ReduceFunction<double[]>() {
                    @Override
                    public double[] reduce(double[] a, double[] b) {
                        double[] r = new double[accLen];
                        for (int i = 0; i < accLen; i++) r[i] = a[i] + b[i];
                        return r;
                    }
                });

        List<double[]> statsList = statsAcc.collect();
        double[] s = statsList.get(0);
        long n = (long) s[accLen - 1];

        // Compute mean-centred covariance matrix C[i][j] = E[xi*xj] - E[xi]*E[xj]
        double[] means = new double[d];
        for (int i = 0; i < d; i++) means[i] = s[i] / n;

        double[][] C = new double[d][d];
        {
            int idx = d;
            for (int i = 0; i < d; i++) {
                for (int j = i; j < d; j++) {
                    double cov = s[idx++] / n - means[i] * means[j];
                    C[i][j] = cov;
                    C[j][i] = cov;
                }
            }
        }

        // --- Driver-side power iteration for top-k eigenvectors ---
        int kActual = Math.min(k, d);
        double[] eigenvalues = new double[kActual];

        for (int comp = 0; comp < kActual; comp++) {
            // Random init unit vector
            double[] v = new double[d];
            {
                Random rng = new Random(comp * 1337L);
                double norm = 0;
                for (int i = 0; i < d; i++) { v[i] = rng.nextGaussian(); norm += v[i]*v[i]; }
                norm = Math.sqrt(norm);
                for (int i = 0; i < d; i++) v[i] /= norm;
            }
            // Power iteration: v = C*v / ||C*v||
            for (int iter = 0; iter < POWER_ITER; iter++) {
                double[] Cv = new double[d];
                for (int i = 0; i < d; i++)
                    for (int j = 0; j < d; j++) Cv[i] += C[i][j] * v[j];
                double norm = 0;
                for (double x : Cv) norm += x * x;
                norm = Math.sqrt(Math.max(norm, 1e-12));
                for (int i = 0; i < d; i++) v[i] = Cv[i] / norm;
            }
            // Eigenvalue = v^T C v
            double[] Cv = new double[d];
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++) Cv[i] += C[i][j] * v[j];
            double lambda = 0;
            for (int i = 0; i < d; i++) lambda += v[i] * Cv[i];
            eigenvalues[comp] = lambda;

            // Deflate: C = C - lambda * v * v^T
            for (int i = 0; i < d; i++)
                for (int j = 0; j < d; j++) C[i][j] -= lambda * v[i] * v[j];
        }

        System.out.println("PCA top-" + kActual + " eigenvalues (n=" + n + ", d=" + d + "):");
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < kActual; i++) {
            System.out.printf("  PC[%d] = %.4f%n", i, eigenvalues[i]);
            sb.append("PC[").append(i).append("]=").append(String.format("%.4f", eigenvalues[i])).append("\n");
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
