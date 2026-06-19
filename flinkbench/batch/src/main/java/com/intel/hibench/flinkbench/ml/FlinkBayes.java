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

import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Flink batch Multinomial Naive Bayes classifier.
 * Generates random dense feature vectors in-memory (avoids reading Spark ML Parquet format).
 * Args: &lt;num_examples&gt; &lt;num_features&gt; &lt;num_classes&gt;
 */
public class FlinkBayes {

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: FlinkBayes <num_examples> <num_features> <num_classes> <output_path>");
            System.exit(1);
        }
        final long numExamples = Long.parseLong(args[0]);
        final int numFeatures  = Integer.parseInt(args[1]);
        final int numClasses   = Integer.parseInt(args[2]);
        final String outputPath = args[3];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Generate random labeled dense feature vectors: (label, double[numFeatures])
        DataSet<Tuple2<Double, double[]>> data = env
                .generateSequence(0, numExamples - 1)
                .mapPartition(new MapPartitionFunction<Long, Tuple2<Double, double[]>>() {
                    @Override
                    public void mapPartition(Iterable<Long> values,
                            Collector<Tuple2<Double, double[]>> out) {
                        Random rng = new Random();
                        for (Long ignored : values) {
                            double label = rng.nextInt(numClasses);
                            double[] features = new double[numFeatures];
                            for (int i = 0; i < numFeatures; i++) {
                                features[i] = Math.abs(rng.nextGaussian()) + 0.1;
                            }
                            out.collect(new Tuple2<>(label, features));
                        }
                    }
                });

        // Split into train (80%) and test (20%) by hashing
        DataSet<Tuple2<Double, double[]>> trainData = data
                .filter(t -> (Math.abs(t.hashCode()) % 10) < 8);
        DataSet<Tuple2<Double, double[]>> testData = data
                .filter(t -> (Math.abs(t.hashCode()) % 10) >= 8);

        // Compute per-class feature sums: (label, featureIdx, sum)
        DataSet<Tuple3<Double, Integer, Double>> featureSums = trainData
                .flatMap(new org.apache.flink.api.common.functions.FlatMapFunction<
                        Tuple2<Double, double[]>, Tuple3<Double, Integer, Double>>() {
                    @Override
                    public void flatMap(Tuple2<Double, double[]> t,
                            Collector<Tuple3<Double, Integer, Double>> out) {
                        for (int i = 0; i < t.f1.length; i++) {
                            out.collect(new Tuple3<>(t.f0, i, t.f1[i]));
                        }
                    }
                })
                .groupBy(0, 1)
                .reduce(new ReduceFunction<Tuple3<Double, Integer, Double>>() {
                    @Override
                    public Tuple3<Double, Integer, Double> reduce(
                            Tuple3<Double, Integer, Double> a,
                            Tuple3<Double, Integer, Double> b) {
                        return new Tuple3<>(a.f0, a.f1, a.f2 + b.f2);
                    }
                });

        // Per-class total feature counts for normalisation
        DataSet<Tuple2<Double, Double>> classTotals = featureSums
                .map(new MapFunction<Tuple3<Double, Integer, Double>, Tuple2<Double, Double>>() {
                    @Override
                    public Tuple2<Double, Double> map(Tuple3<Double, Integer, Double> t) {
                        return new Tuple2<>(t.f0, t.f2);
                    }
                })
                .groupBy(0)
                .reduce(new ReduceFunction<Tuple2<Double, Double>>() {
                    @Override
                    public Tuple2<Double, Double> reduce(Tuple2<Double, Double> a,
                            Tuple2<Double, Double> b) {
                        return new Tuple2<>(a.f0, a.f1 + b.f1);
                    }
                });

        // Per-class example counts for priors
        DataSet<Tuple2<Double, Long>> classCounts = trainData
                .map(new MapFunction<Tuple2<Double, double[]>, Tuple2<Double, Long>>() {
                    @Override
                    public Tuple2<Double, Long> map(Tuple2<Double, double[]> t) {
                        return new Tuple2<>(t.f0, 1L);
                    }
                })
                .groupBy(0)
                .reduce(new ReduceFunction<Tuple2<Double, Long>>() {
                    @Override
                    public Tuple2<Double, Long> reduce(Tuple2<Double, Long> a,
                            Tuple2<Double, Long> b) {
                        return new Tuple2<>(a.f0, a.f1 + b.f1);
                    }
                });

        long totalTrain = FlinkJobUtils.count(trainData, env, "NaiveBayes (train count)");

        // Classify test examples using broadcast model
        DataSet<Tuple2<Double, Double>> predictions = testData
                .map(new ClassifyNaiveBayes())
                .withBroadcastSet(featureSums, "featureSums")
                .withBroadcastSet(classTotals, "classTotals")
                .withBroadcastSet(classCounts, "classCounts");

        // Compute accuracy
        DataSet<Double> correctCount = predictions
                .map(new MapFunction<Tuple2<Double, Double>, Double>() {
                    @Override
                    public Double map(Tuple2<Double, Double> t) {
                        return t.f0.equals(t.f1) ? 1.0 : 0.0;
                    }
                })
                .reduce(new ReduceFunction<Double>() {
                    @Override
                    public Double reduce(Double a, Double b) { return a + b; }
                });

        long testCount = FlinkJobUtils.count(testData, env, "NaiveBayes (test count)");
        List<Double> correctList = FlinkJobUtils.collect(correctCount, env, "NaiveBayes");
        double accuracy = testCount == 0 ? 0.0 : correctList.get(0) / testCount;
        System.out.println("Naive Bayes accuracy: " + String.format("%.4f", accuracy)
                + " (trained on " + totalTrain + " examples)");
        writeResult(outputPath, "NaiveBayes accuracy=" + String.format("%.4f", accuracy) + "\n");
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

    public static class ClassifyNaiveBayes
            extends RichMapFunction<Tuple2<Double, double[]>, Tuple2<Double, Double>> {

        private Map<String, Double> logProbs;
        private Map<Double, Double> logPriors;

        @Override
        public void open(Configuration parameters) {
            Collection<Tuple3<Double, Integer, Double>> featureSums =
                    getRuntimeContext().getBroadcastVariable("featureSums");
            Collection<Tuple2<Double, Double>> classTotals =
                    getRuntimeContext().getBroadcastVariable("classTotals");
            Collection<Tuple2<Double, Long>> classCounts =
                    getRuntimeContext().getBroadcastVariable("classCounts");

            Map<Double, Double> totalMap = new HashMap<>();
            for (Tuple2<Double, Double> t : classTotals) totalMap.put(t.f0, t.f1);

            logProbs = new HashMap<>();
            for (Tuple3<Double, Integer, Double> t : featureSums) {
                double classTotal = totalMap.getOrDefault(t.f0, 1.0);
                logProbs.put(t.f0 + "_" + t.f1,
                        Math.log((t.f2 + 1.0) / (classTotal + 1.0)));
            }

            long totalCount = 0;
            Map<Double, Long> countMap = new HashMap<>();
            for (Tuple2<Double, Long> t : classCounts) {
                countMap.put(t.f0, t.f1);
                totalCount += t.f1;
            }
            logPriors = new HashMap<>();
            final long total = totalCount;
            for (Map.Entry<Double, Long> e : countMap.entrySet()) {
                logPriors.put(e.getKey(), Math.log((double) e.getValue() / total));
            }
        }

        @Override
        public Tuple2<Double, Double> map(Tuple2<Double, double[]> example) {
            double bestLabel = -1;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (Map.Entry<Double, Double> prior : logPriors.entrySet()) {
                double label = prior.getKey();
                double score = prior.getValue();
                for (int i = 0; i < example.f1.length; i++) {
                    double lp = logProbs.getOrDefault(label + "_" + i, -10.0);
                    score += example.f1[i] * lp;
                }
                if (score > bestScore) { bestScore = score; bestLabel = label; }
            }
            return new Tuple2<>(example.f0, bestLabel);
        }
    }
}
