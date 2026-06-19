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
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Flink batch Alternating Least Squares (ALS) for collaborative filtering.
 * Generates random ratings in-memory (avoids reading Spark ML SequenceFile format).
 * Args: &lt;num_users&gt; &lt;num_products&gt; &lt;num_ratings&gt; &lt;rank&gt; &lt;max_iterations&gt; &lt;lambda&gt;
 */
public class FlinkALS {

    public static void main(String[] args) throws Exception {
        if (args.length < 7) {
            System.err.println(
                "Usage: FlinkALS <num_users> <num_products> <num_ratings> <rank> <max_iterations> <lambda> <output_path>");
            System.exit(1);
        }
        final int numUsers      = Integer.parseInt(args[0]);
        final int numProducts   = Integer.parseInt(args[1]);
        final long numRatingsL  = Long.parseLong(args[2]);
        final int rank          = Integer.parseInt(args[3]);
        final int maxIterations = Integer.parseInt(args[4]);
        final double lambda     = Double.parseDouble(args[5]);
        final String outputPath = args[6];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Generate random ratings (userId, productId, rating)
        DataSet<Tuple3<Integer, Integer, Double>> ratings = env
                .generateSequence(0, numRatingsL - 1)
                .mapPartition(new MapPartitionFunction<Long, Tuple3<Integer, Integer, Double>>() {
                    @Override
                    public void mapPartition(Iterable<Long> values,
                            org.apache.flink.util.Collector<Tuple3<Integer, Integer, Double>> out) {
                        Random rng = new Random();
                        for (Long ignored : values) {
                            out.collect(new Tuple3<>(
                                rng.nextInt(numUsers),
                                rng.nextInt(numProducts),
                                1.0 + rng.nextDouble() * 4.0));
                        }
                    }
                });

        // Initialise user factors randomly: (userId, double[rank])
        DataSet<Tuple2<Integer, double[]>> userFactors = env
                .generateSequence(0, numUsers - 1)
                .map(new MapFunction<Long, Tuple2<Integer, double[]>>() {
                    @Override
                    public Tuple2<Integer, double[]> map(Long id) {
                        Random rng = new Random(id);
                        double[] factors = new double[rank];
                        for (int i = 0; i < rank; i++) factors[i] = rng.nextGaussian() * 0.1;
                        return new Tuple2<>((int)(long) id, factors);
                    }
                });

        // Initialise product factors randomly: (productId, double[rank])
        DataSet<Tuple2<Integer, double[]>> productFactors = env
                .generateSequence(0, numProducts - 1)
                .map(new MapFunction<Long, Tuple2<Integer, double[]>>() {
                    @Override
                    public Tuple2<Integer, double[]> map(Long id) {
                        Random rng = new Random(id + 100000L);
                        double[] factors = new double[rank];
                        for (int i = 0; i < rank; i++) factors[i] = rng.nextGaussian() * 0.1;
                        return new Tuple2<>((int)(long) id, factors);
                    }
                });

        // ALS iterations (manual loop; Flink IterativeDataSet works for one dataset,
        // but ALS alternates two datasets — we implement a fixed-count outer loop)
        for (int iter = 0; iter < maxIterations; iter++) {
            // Fix product factors, update user factors
            // For each user: solve (P^T P + lambda*I) u = P^T r_u
            userFactors = updateFactors(ratings, productFactors, rank, lambda, true);

            // Fix user factors, update product factors
            productFactors = updateFactors(ratings, userFactors, rank, lambda, false);
        }

        // Compute RMSE on training data
        final DataSet<Tuple2<Integer, double[]>> finalUserFactors = userFactors;
        final DataSet<Tuple2<Integer, double[]>> finalProductFactors = productFactors;

        DataSet<Double> squaredErrors = ratings
                .join(finalUserFactors).where(0).equalTo(0)
                .map(new MapFunction<Tuple2<Tuple3<Integer, Integer, Double>, Tuple2<Integer, double[]>>,
                        Tuple3<Integer, Integer, Double>>() {
                    @Override
                    public Tuple3<Integer, Integer, Double> map(
                            Tuple2<Tuple3<Integer, Integer, Double>, Tuple2<Integer, double[]>> t) {
                        // (productId, userId, rating * user_factor combined) — carry user factor
                        // We need to join with product factor next; pack user factor in rating field by encoding
                        // Workaround: emit (productId, userId, rating) keeping same structure and
                        // handle the dot-product in a second join
                        return t.f0; // pass through for now — handled below
                    }
                })
                .join(finalProductFactors).where(1).equalTo(0)
                .map(new MapFunction<Tuple2<Tuple3<Integer, Integer, Double>, Tuple2<Integer, double[]>>, Double>() {
                    @Override
                    public Double map(
                            Tuple2<Tuple3<Integer, Integer, Double>, Tuple2<Integer, double[]>> joined) {
                        // We need user factor too — this simplified version just returns 0 for RMSE placeholder
                        return 0.0;
                    }
                });

        // Compute proper RMSE using RichMapFunction with broadcast
        DataSet<Double> rmseDs = ratings
                .map(new ComputeRMSE(rank))
                .withBroadcastSet(finalUserFactors, "userFactors")
                .withBroadcastSet(finalProductFactors, "productFactors")
                .reduce(new ReduceFunction<Double>() {
                    @Override
                    public Double reduce(Double a, Double b) {
                        return a + b;
                    }
                });

        List<Double> rmseList = FlinkJobUtils.collect(rmseDs, env, "ALS");
        double rmse = rmseList.isEmpty() ? 0.0 : Math.sqrt(rmseList.get(0) / numRatingsL);
        System.out.println("ALS training complete. RMSE = " + String.format("%.4f", rmse));
        writeResult(outputPath, "ALS RMSE=" + String.format("%.4f", rmse) + "\n");

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

    /**
     * Updates one set of factors by solving the least-squares system.
     * If updateUsers==true, updates user factors using fixed product factors.
     * Uses a RichMapFunction with broadcast of the fixed factors and ratings.
     */
    private static DataSet<Tuple2<Integer, double[]>> updateFactors(
            DataSet<Tuple3<Integer, Integer, Double>> ratings,
            DataSet<Tuple2<Integer, double[]>> fixedFactors,
            final int rank,
            final double lambda,
            final boolean updateUsers) {

        // Group ratings by the entity being updated (user or product)
        // field 0 = userId, field 1 = productId
        int groupField = updateUsers ? 0 : 1;
        int fixedField = updateUsers ? 1 : 0;

        // Join ratings with fixed factors on the fixed entity
        DataSet<Tuple3<Integer, Double, double[]>> ratingWithFixed = ratings
                .join(fixedFactors).where(fixedField).equalTo(0)
                .map(new MapFunction<Tuple2<Tuple3<Integer, Integer, Double>, Tuple2<Integer, double[]>>,
                        Tuple3<Integer, Double, double[]>>() {
                    @Override
                    public Tuple3<Integer, Double, double[]> map(
                            Tuple2<Tuple3<Integer, Integer, Double>, Tuple2<Integer, double[]>> joined) {
                        Tuple3<Integer, Integer, Double> rating = joined.f0;
                        Tuple2<Integer, double[]> factor = joined.f1;
                        int updateId = updateUsers ? rating.f0 : rating.f1;
                        return new Tuple3<>(updateId, rating.f2, factor.f1);
                    }
                });

        // Group by update entity id, accumulate ATA and ATb, then solve
        return ratingWithFixed
                .groupBy(0)
                .reduceGroup(new org.apache.flink.api.common.functions.GroupReduceFunction<
                        Tuple3<Integer, Double, double[]>,
                        Tuple2<Integer, double[]>>() {
                    @Override
                    public void reduce(Iterable<Tuple3<Integer, Double, double[]>> group,
                            Collector<Tuple2<Integer, double[]>> out) {
                        List<Tuple3<Integer, Double, double[]>> items = new ArrayList<>();
                        int entityId = -1;
                        for (Tuple3<Integer, Double, double[]> item : group) {
                            items.add(new Tuple3<>(item.f0, item.f1, item.f2.clone()));
                            entityId = item.f0;
                        }
                        if (items.isEmpty()) return;

                        // Build ATA and ATb
                        double[] AtA = new double[rank * rank];
                        double[] Atb = new double[rank];
                        for (Tuple3<Integer, Double, double[]> item : items) {
                            double[] p = item.f2;
                            double r = item.f1;
                            for (int i = 0; i < rank; i++) {
                                Atb[i] += p[i] * r;
                                for (int j = 0; j < rank; j++) {
                                    AtA[i * rank + j] += p[i] * p[j];
                                }
                            }
                        }
                        // Add regularisation: AtA += lambda * I
                        for (int i = 0; i < rank; i++) {
                            AtA[i * rank + i] += lambda;
                        }
                        // Solve AtA * x = Atb using Cholesky / Gaussian elimination
                        double[] factors = solveLinear(AtA, Atb, rank);
                        out.collect(new Tuple2<>(entityId, factors));
                    }
                });
    }

    /** Simple Gaussian elimination to solve Ax = b. */
    private static double[] solveLinear(double[] flatA, double[] b, int n) {
        double[][] A = new double[n][n];
        double[] x = b.clone();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = flatA[i * n + j];
            }
        }
        for (int col = 0; col < n; col++) {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(A[row][col]) > Math.abs(A[maxRow][col])) maxRow = row;
            }
            double[] tmp = A[col]; A[col] = A[maxRow]; A[maxRow] = tmp;
            double tb = x[col]; x[col] = x[maxRow]; x[maxRow] = tb;

            if (Math.abs(A[col][col]) < 1e-12) continue;
            for (int row = col + 1; row < n; row++) {
                double factor = A[row][col] / A[col][col];
                for (int k = col; k < n; k++) {
                    A[row][k] -= factor * A[col][k];
                }
                x[row] -= factor * x[col];
            }
        }
        // Back substitution
        for (int i = n - 1; i >= 0; i--) {
            if (Math.abs(A[i][i]) < 1e-12) { x[i] = 0; continue; }
            x[i] /= A[i][i];
            for (int k = i - 1; k >= 0; k--) {
                x[k] -= A[k][i] * x[i];
            }
        }
        return x;
    }

    /** Computes squared error for a single rating using broadcast factors. */
    public static class ComputeRMSE
            extends RichMapFunction<Tuple3<Integer, Integer, Double>, Double> {

        private final int rank;
        private Map<Integer, double[]> userMap;
        private Map<Integer, double[]> productMap;

        public ComputeRMSE(int rank) {
            this.rank = rank;
        }

        @Override
        public void open(Configuration parameters) {
            Collection<Tuple2<Integer, double[]>> uFactors =
                    getRuntimeContext().getBroadcastVariable("userFactors");
            Collection<Tuple2<Integer, double[]>> pFactors =
                    getRuntimeContext().getBroadcastVariable("productFactors");
            userMap = new HashMap<>();
            productMap = new HashMap<>();
            for (Tuple2<Integer, double[]> t : uFactors) userMap.put(t.f0, t.f1);
            for (Tuple2<Integer, double[]> t : pFactors) productMap.put(t.f0, t.f1);
        }

        @Override
        public Double map(Tuple3<Integer, Integer, Double> rating) {
            double[] u = userMap.get(rating.f0);
            double[] p = productMap.get(rating.f1);
            if (u == null || p == null) return 0.0;
            double dot = 0;
            for (int i = 0; i < rank; i++) dot += u[i] * p[i];
            double err = dot - rating.f2;
            return err * err;
        }
    }
}
