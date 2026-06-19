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

import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.common.functions.RichMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.operators.IterativeDataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;

import java.util.Collection;
import java.util.List;
import java.util.Random;

/**
 * Flink batch Logistic Regression via gradient descent with IterativeDataSet.
 * Generates binary labeled data and trains a logistic classifier.
 * Args: &lt;input_path&gt; &lt;num_examples&gt; &lt;num_features&gt;
 */
public class FlinkLogisticRegression {

    private static final int MAX_ITERATIONS = 100;
    private static final double LEARNING_RATE = 0.1;

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println(
                "Usage: FlinkLogisticRegression <input_path> <num_examples> <num_features> <output_path>");
            System.exit(1);
        }
        final long numExamples = Long.parseLong(args[1]);
        final int numFeatures = Integer.parseInt(args[2]);
        final String outputPath = args[3];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // True weights for data generation
        final double[] trueWeights = new double[numFeatures];
        {
            Random rng = new Random(7);
            for (int i = 0; i < numFeatures; i++) trueWeights[i] = rng.nextGaussian();
        }

        // Generate binary labeled data: label = Bernoulli(sigmoid(dot(w, x)))
        DataSet<Tuple2<double[], Double>> data = env.generateSequence(0, numExamples - 1)
                .mapPartition(new MapPartitionFunction<Long, Tuple2<double[], Double>>() {
                    @Override
                    public void mapPartition(Iterable<Long> values,
                            Collector<Tuple2<double[], Double>> out) {
                        Random rng = new Random();
                        for (Long seed : values) {
                            Random r2 = new Random(seed);
                            double[] features = new double[numFeatures];
                            for (int i = 0; i < numFeatures; i++) features[i] = r2.nextGaussian();
                            double logit = 0;
                            for (int i = 0; i < numFeatures; i++) logit += trueWeights[i] * features[i];
                            double prob = 1.0 / (1.0 + Math.exp(-logit));
                            double label = rng.nextDouble() < prob ? 1.0 : 0.0;
                            out.collect(new Tuple2<>(features, label));
                        }
                    }
                });

        // Initialise weights to zero
        DataSet<double[]> weights = env.fromElements(new double[numFeatures]);

        IterativeDataSet<double[]> loop = weights.iterate(MAX_ITERATIONS);

        // Compute gradient of log-loss: grad = (1/n) * sum_i (sigmoid(w^T x_i) - y_i) * x_i
        DataSet<double[]> gradient = data
                .map(new LogisticGradient(numFeatures))
                .withBroadcastSet(loop, "weights")
                .reduce(new ReduceFunction<double[]>() {
                    @Override
                    public double[] reduce(double[] a, double[] b) {
                        double[] result = new double[a.length];
                        for (int i = 0; i < a.length; i++) result[i] = a[i] + b[i];
                        return result;
                    }
                });

        // Update: w = w - lr * (1/n) * gradSum
        DataSet<double[]> newWeights = gradient
                .map(new RichMapFunction<double[], double[]>() {
                    private Collection<double[]> wColl;

                    @Override
                    public void open(Configuration parameters) {
                        wColl = getRuntimeContext().getBroadcastVariable("weights");
                    }

                    @Override
                    public double[] map(double[] gradSum) {
                        double[] w = wColl.iterator().next();
                        double[] newW = new double[numFeatures];
                        double scale = LEARNING_RATE / numExamples;
                        for (int i = 0; i < numFeatures; i++) {
                            newW[i] = w[i] - scale * gradSum[i];
                        }
                        return newW;
                    }
                })
                .withBroadcastSet(loop, "weights");

        DataSet<double[]> finalWeights = loop.closeWith(newWeights);

        // Compute training accuracy
        DataSet<double[]> accuracy = data
                .map(new RichMapFunction<Tuple2<double[], Double>, double[]>() {
                    private Collection<double[]> wColl;

                    @Override
                    public void open(Configuration parameters) {
                        wColl = getRuntimeContext().getBroadcastVariable("weights");
                    }

                    @Override
                    public double[] map(Tuple2<double[], Double> example) {
                        double[] w = wColl.iterator().next();
                        double logit = 0;
                        for (int i = 0; i < numFeatures; i++) logit += w[i] * example.f0[i];
                        double pred = logit >= 0 ? 1.0 : 0.0;
                        return new double[]{pred == example.f1 ? 1.0 : 0.0, 1.0};
                    }
                })
                .withBroadcastSet(finalWeights, "weights")
                .reduce(new ReduceFunction<double[]>() {
                    @Override
                    public double[] reduce(double[] a, double[] b) {
                        return new double[]{a[0] + b[0], a[1] + b[1]};
                    }
                });

        List<double[]> accList = FlinkJobUtils.collect(accuracy, env, "LogisticRegression");
        double acc = accList.isEmpty() ? 0.0 : accList.get(0)[0] / accList.get(0)[1];
        System.out.println("Logistic Regression training accuracy: " + String.format("%.4f", acc));
        writeResult(outputPath, "LogisticRegression accuracy=" + String.format("%.4f", acc) + "\n");

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

    public static class LogisticGradient extends RichMapFunction<Tuple2<double[], Double>, double[]> {

        private final int numFeatures;
        private Collection<double[]> weightsCollection;

        public LogisticGradient(int numFeatures) {
            this.numFeatures = numFeatures;
        }

        @Override
        public void open(Configuration parameters) {
            weightsCollection = getRuntimeContext().getBroadcastVariable("weights");
        }

        @Override
        public double[] map(Tuple2<double[], Double> example) {
            double[] w = weightsCollection.iterator().next();
            double[] x = example.f0;
            double y = example.f1;
            double logit = 0;
            for (int i = 0; i < numFeatures; i++) logit += w[i] * x[i];
            double pred = 1.0 / (1.0 + Math.exp(-logit));
            double err = pred - y;
            double[] grad = new double[numFeatures];
            for (int i = 0; i < numFeatures; i++) grad[i] = err * x[i];
            return grad;
        }
    }
}
