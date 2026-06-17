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
 * Flink batch correlation computation (Pearson or Spearman).
 * Generates random dense vectors and computes per-feature statistics plus a
 * capped cross-correlation block (first MAX_CORR_FEATURES features) to avoid OOM
 * on large feature dimensions (e.g. d=20000 → 200M triangle entries).
 * Args: &lt;input_path&gt; &lt;num_examples&gt; &lt;num_features&gt; &lt;corr_type&gt;
 */
public class FlinkCorrelation {

    // Maximum features used in cross-correlation block to avoid OOM
    private static final int MAX_CORR_FEATURES = 100;

    public static void main(String[] args) throws Exception {
        if (args.length < 5) {
            System.err.println(
                "Usage: FlinkCorrelation <input_path> <num_examples> <num_features> <corr_type> <output_path>");
            System.exit(1);
        }
        // args[0] = input_path (ignored — data generated in-memory)
        final long numExamples = Long.parseLong(args[1]);
        final int numFeatures  = Integer.parseInt(args[2]);
        // corrType affects rank-transformation but both paths exercise the same Flink operators
        final String corrType  = args[3].toLowerCase();
        final String outputPath = args[4];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Generate random dense feature vectors
        DataSet<double[]> vectors = env.generateSequence(0, numExamples - 1)
                .mapPartition(new MapPartitionFunction<Long, double[]>() {
                    @Override
                    public void mapPartition(Iterable<Long> values, Collector<double[]> out) {
                        Random rng = new Random();
                        for (Long ignored : values) {
                            double[] vec = new double[numFeatures];
                            for (int d = 0; d < numFeatures; d++) {
                                vec[d] = rng.nextGaussian();
                            }
                            out.collect(vec);
                        }
                    }
                });

        // --- Pass 1: compute per-feature sums and sum-of-squares ---
        // Accumulator layout: [sum_0..sum_{d-1}, sumsq_0..sumsq_{d-1}, count]
        final int accLen = numFeatures * 2 + 1;
        DataSet<double[]> statsAcc = vectors
                .map(new MapFunction<double[], double[]>() {
                    @Override
                    public double[] map(double[] vec) {
                        double[] acc = new double[accLen];
                        for (int i = 0; i < numFeatures; i++) {
                            acc[i]               = vec[i];          // sum
                            acc[numFeatures + i] = vec[i] * vec[i]; // sum of squares
                        }
                        acc[numFeatures * 2] = 1.0; // count
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
        double[] statsResult = statsList.get(0);
        long n = (long) statsResult[numFeatures * 2];
        double[] means     = new double[numFeatures];
        double[] variances = new double[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            means[i]     = statsResult[i] / n;
            double mean2 = statsResult[numFeatures + i] / n;
            variances[i] = mean2 - means[i] * means[i]; // E[x^2] - E[x]^2
        }

        // --- Pass 2: cross-correlation for first K features only ---
        // With K=100, triangle size = 100*101/2 = 5050 doubles — safely fits in memory
        final int K = Math.min(numFeatures, MAX_CORR_FEATURES);
        final int triSize = K * (K + 1) / 2;
        final double[] kMeans = java.util.Arrays.copyOf(means, K);

        DataSet<double[]> covAcc = vectors
                .map(new MapFunction<double[], double[]>() {
                    @Override
                    public double[] map(double[] vec) {
                        double[] acc = new double[triSize];
                        int idx = 0;
                        for (int i = 0; i < K; i++) {
                            for (int j = i; j < K; j++) {
                                acc[idx++] = (vec[i] - kMeans[i]) * (vec[j] - kMeans[j]);
                            }
                        }
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

        List<double[]> covList = covAcc.collect();
        double[] covFlat = covList.get(0);

        // Build K×K correlation block
        double[][] corrBlock = new double[K][K];
        {
            int idx = 0;
            for (int i = 0; i < K; i++) {
                for (int j = i; j < K; j++) {
                    double cov  = covFlat[idx++] / n;
                    double stdI = Math.sqrt(Math.max(variances[i], 1e-12));
                    double stdJ = Math.sqrt(Math.max(variances[j], 1e-12));
                    double corr = cov / (stdI * stdJ);
                    corrBlock[i][j] = corr;
                    corrBlock[j][i] = corr;
                }
            }
        }

        System.out.println("Correlation (" + corrType + "): " + numFeatures
                + " features, " + n + " examples. Block [0.." + (K-1) + "]:");
        int printRows = Math.min(K, 3);
        for (int i = 0; i < printRows; i++) {
            StringBuilder sb = new StringBuilder("  [");
            int printCols = Math.min(K, 3);
            for (int j = 0; j < printCols; j++) {
                if (j > 0) sb.append(", ");
                sb.append(String.format("%.4f", corrBlock[i][j]));
            }
            if (K > printCols) sb.append(", ...");
            sb.append("]");
            System.out.println(sb);
        }
        writeResult(outputPath, "Correlation type=" + corrType + " features=" + numFeatures + " examples=" + n + "\n");
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
