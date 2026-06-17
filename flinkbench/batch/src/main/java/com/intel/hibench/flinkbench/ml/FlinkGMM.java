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
import org.apache.flink.api.java.operators.IterativeDataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.api.java.utils.DataSetUtils;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

public class FlinkGMM {

    public static class GaussianComponent implements Serializable {
        public int id;
        public double weight;
        public double[] mean;
        public double variance;

        public GaussianComponent() {}

        public GaussianComponent(int id, double weight, double[] mean, double variance) {
            this.id = id;
            this.weight = weight;
            this.mean = mean;
            this.variance = variance;
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 5) {
            System.err.println("Usage: FlinkGMM <num_examples> <num_features> <k> <max_iterations> <output_path>");
            System.exit(1);
        }
        final long numExamples   = Long.parseLong(args[0]);
        final int numFeatures    = Integer.parseInt(args[1]);
        final int k              = Integer.parseInt(args[2]);
        final int maxIterations  = Integer.parseInt(args[3]);
        final String outputPath  = args[4];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        DataSet<double[]> points = env.generateSequence(0, numExamples - 1)
            .mapPartition(new MapPartitionFunction<Long, double[]>() {
                @Override
                public void mapPartition(Iterable<Long> values, Collector<double[]> out) {
                    Random rng = new Random();
                    for (Long id : values) {
                        double[] v = new double[numFeatures];
                        for (int i = 0; i < numFeatures; i++) v[i] = rng.nextGaussian();
                        out.collect(v);
                    }
                }
            });

        // Initialise: pick first k points as means, equal weights, unit variance
        List<double[]> seeds = DataSetUtils.zipWithIndex(points.first(k))
            .map(new MapFunction<Tuple2<Long, double[]>, double[]>() {
                @Override
                public double[] map(Tuple2<Long, double[]> t) { return t.f1; }
            }).collect();

        List<GaussianComponent> initComponents = new ArrayList<>();
        for (int i = 0; i < seeds.size(); i++) {
            initComponents.add(new GaussianComponent(i, 1.0 / k, seeds.get(i), 1.0));
        }

        DataSet<GaussianComponent> components = env.fromCollection(initComponents);
        IterativeDataSet<GaussianComponent> loop = components.iterate(maxIterations);

        DataSet<Tuple3<Integer, double[], Double>> stats = points
            .map(new ComputeStats(numFeatures))
            .withBroadcastSet(loop, "components")
            .groupBy(0)
            .reduce(new ReduceFunction<Tuple3<Integer, double[], Double>>() {
                @Override
                public Tuple3<Integer, double[], Double> reduce(
                        Tuple3<Integer, double[], Double> a,
                        Tuple3<Integer, double[], Double> b) {
                    double[] s = new double[a.f1.length];
                    for (int i = 0; i < s.length; i++) s[i] = a.f1[i] + b.f1[i];
                    return new Tuple3<>(a.f0, s, a.f2 + b.f2);
                }
            });

        DataSet<Double> totalR = stats
            .map(new MapFunction<Tuple3<Integer, double[], Double>, Double>() {
                @Override
                public Double map(Tuple3<Integer, double[], Double> t) { return t.f1[0]; }
            })
            .reduce(new ReduceFunction<Double>() {
                @Override
                public Double reduce(Double a, Double b) { return a + b; }
            });

        DataSet<GaussianComponent> newComponents = stats
            .map(new UpdateComponents(numFeatures))
            .withBroadcastSet(totalR, "totalR");

        DataSet<GaussianComponent> finalComponents = loop.closeWith(newComponents);

        List<GaussianComponent> finalList = finalComponents.collect();
        StringBuilder sb = new StringBuilder();
        for (GaussianComponent c : finalList) {
            sb.append("component=").append(c.id)
              .append(" weight=").append(String.format("%.4f", c.weight))
              .append(" variance=").append(String.format("%.4f", c.variance))
              .append("\n");
        }
        System.out.println("GMM complete (" + maxIterations + " iterations, k=" + k + ")");
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

    public static class ComputeStats
            extends RichMapFunction<double[], Tuple3<Integer, double[], Double>> {

        private final int dim;
        private Collection<GaussianComponent> components;

        public ComputeStats(int dim) { this.dim = dim; }

        @Override
        public void open(Configuration parameters) {
            this.components = getRuntimeContext().getBroadcastVariable("components");
        }

        @Override
        public Tuple3<Integer, double[], Double> map(double[] point) {
            List<GaussianComponent> list = new ArrayList<>(components);
            double[] logProbs = new double[list.size()];
            for (int i = 0; i < list.size(); i++) {
                GaussianComponent c = list.get(i);
                double distSq = 0;
                for (int d = 0; d < point.length; d++) {
                    double diff = point[d] - c.mean[d];
                    distSq += diff * diff;
                }
                double sigma2 = Math.max(c.variance, 1e-10);
                logProbs[i] = Math.log(Math.max(c.weight, 1e-300))
                    - 0.5 * point.length * Math.log(2 * Math.PI * sigma2)
                    - distSq / (2 * sigma2);
            }
            double maxLog = Double.NEGATIVE_INFINITY;
            for (double lp : logProbs) if (lp > maxLog) maxLog = lp;
            double[] r = new double[logProbs.length];
            double sumR = 0;
            for (int i = 0; i < logProbs.length; i++) { r[i] = Math.exp(logProbs[i] - maxLog); sumR += r[i]; }
            for (int i = 0; i < r.length; i++) r[i] /= sumR;

            int best = 0;
            for (int i = 1; i < r.length; i++) if (r[i] > r[best]) best = i;

            double[] packed = new double[1 + point.length];
            packed[0] = r[best];
            double varContrib = 0;
            for (int d = 0; d < point.length; d++) {
                packed[1 + d] = r[best] * point[d];
                double diff = point[d] - list.get(best).mean[d];
                varContrib += r[best] * diff * diff;
            }
            return new Tuple3<>(list.get(best).id, packed, varContrib);
        }
    }

    public static class UpdateComponents
            extends RichMapFunction<Tuple3<Integer, double[], Double>, GaussianComponent> {

        private final int dim;
        private Collection<Double> totalRCollection;

        public UpdateComponents(int dim) { this.dim = dim; }

        @Override
        public void open(Configuration parameters) {
            this.totalRCollection = getRuntimeContext().getBroadcastVariable("totalR");
        }

        @Override
        public GaussianComponent map(Tuple3<Integer, double[], Double> stats) {
            double totalR = totalRCollection.iterator().next();
            double sumR = stats.f1[0];
            double weight = sumR / Math.max(totalR, 1e-300);
            double[] mean = new double[stats.f1.length - 1];
            for (int d = 0; d < mean.length; d++) mean[d] = stats.f1[1 + d] / Math.max(sumR, 1e-300);
            double variance = Math.max(stats.f2 / Math.max(sumR * mean.length, 1e-300), 1e-6);
            return new GaussianComponent(stats.f0, weight, mean, variance);
        }
    }
}
