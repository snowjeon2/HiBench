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

import java.util.Collection;
import java.util.List;
import java.util.Random;

public class FlinkKMeans {

    public static void main(String[] args) throws Exception {
        if (args.length < 5) {
            System.err.println(
                "Usage: FlinkKMeans <num_examples> <num_features> <k> <max_iterations> <output_path>");
            System.exit(1);
        }
        final long numExamples = Long.parseLong(args[0]);
        final int numFeatures  = Integer.parseInt(args[1]);
        final int k            = Integer.parseInt(args[2]);
        final int maxIterations = Integer.parseInt(args[3]);
        final String outputPath = args[4];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Generate random points (numExamples x numFeatures)
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

        // Pick first k points as initial centroids via DataSetUtils.zipWithIndex
        DataSet<Tuple2<Integer, double[]>> centroids = DataSetUtils
            .zipWithIndex(points.first(k))
            .map(new MapFunction<Tuple2<Long, double[]>, Tuple2<Integer, double[]>>() {
                @Override
                public Tuple2<Integer, double[]> map(Tuple2<Long, double[]> t) {
                    return new Tuple2<>((int)(long) t.f0, t.f1);
                }
            });

        // KMeans iteration
        IterativeDataSet<Tuple2<Integer, double[]>> loop = centroids.iterate(maxIterations);

        DataSet<Tuple2<Integer, double[]>> assigned = points
            .map(new SelectNearestCentroid())
            .withBroadcastSet(loop, "centroids");

        DataSet<Tuple2<Integer, double[]>> newCentroids = assigned
            .map(new MapFunction<Tuple2<Integer, double[]>, Tuple3<Integer, double[], Long>>() {
                @Override
                public Tuple3<Integer, double[], Long> map(Tuple2<Integer, double[]> t) {
                    return new Tuple3<>(t.f0, t.f1, 1L);
                }
            })
            .groupBy(0)
            .reduce(new ReduceFunction<Tuple3<Integer, double[], Long>>() {
                @Override
                public Tuple3<Integer, double[], Long> reduce(
                        Tuple3<Integer, double[], Long> a,
                        Tuple3<Integer, double[], Long> b) {
                    double[] sum = new double[a.f1.length];
                    for (int i = 0; i < sum.length; i++) sum[i] = a.f1[i] + b.f1[i];
                    return new Tuple3<>(a.f0, sum, a.f2 + b.f2);
                }
            })
            .map(new MapFunction<Tuple3<Integer, double[], Long>, Tuple2<Integer, double[]>>() {
                @Override
                public Tuple2<Integer, double[]> map(Tuple3<Integer, double[], Long> t) {
                    double[] mean = new double[t.f1.length];
                    for (int i = 0; i < mean.length; i++) mean[i] = t.f1[i] / t.f2;
                    return new Tuple2<>(t.f0, mean);
                }
            });

        DataSet<Tuple2<Integer, double[]>> finalCentroids = loop.closeWith(newCentroids);

        List<Tuple2<Integer, double[]>> finalList = finalCentroids.collect();
        StringBuilder sb = new StringBuilder();
        for (Tuple2<Integer, double[]> c : finalList) {
            sb.append("centroid=").append(c.f0).append(" mean=[");
            int show = Math.min(c.f1.length, 3);
            for (int i = 0; i < show; i++) {
                if (i > 0) sb.append(", ");
                sb.append(String.format("%.4f", c.f1[i]));
            }
            if (c.f1.length > show) sb.append(", ...");
            sb.append("]\n");
        }
        System.out.println("KMeans complete (" + maxIterations + " iterations, k=" + k + ")");
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

    public static class SelectNearestCentroid
            extends RichMapFunction<double[], Tuple2<Integer, double[]>> {

        private Collection<Tuple2<Integer, double[]>> centroids;

        @Override
        public void open(Configuration parameters) {
            this.centroids = getRuntimeContext().getBroadcastVariable("centroids");
        }

        @Override
        public Tuple2<Integer, double[]> map(double[] point) {
            int nearestId = -1;
            double minDist = Double.MAX_VALUE;
            for (Tuple2<Integer, double[]> c : centroids) {
                double dist = 0;
                for (int i = 0; i < point.length; i++) {
                    double d = point[i] - c.f1[i];
                    dist += d * d;
                }
                if (dist < minDist) { minDist = dist; nearestId = c.f0; }
            }
            return new Tuple2<>(nearestId, point);
        }
    }
}
