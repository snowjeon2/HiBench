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
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;

import java.util.Collection;
import java.util.List;
import java.util.Random;

/**
 * Flink batch Linear Regression via mini-batch gradient descent.
 * Generates synthetic labeled data and trains using IterativeDataSet.
 * Args: &lt;input_path&gt; &lt;num_examples&gt; &lt;num_features&gt; &lt;max_iterations&gt; &lt;reg_param&gt;
 */
public class FlinkLinearRegression {

    public static void main(String[] args) throws Exception {
        if (args.length < 6) {
            System.err.println(
                "Usage: FlinkLinearRegression <input_path> <num_examples> <num_features> "
                + "<max_iterations> <reg_param> <output_path>");
            System.exit(1);
        }
        final long numExamples = Long.parseLong(args[1]);
        final int numFeatures = Integer.parseInt(args[2]);
        final int maxIterations = Integer.parseInt(args[3]);
        final double regParam = Double.parseDouble(args[4]);
        final String outputPath = args[5];
        final double learningRate = 0.01;

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // True weights used to generate data (fixed seed for reproducibility)
        final double[] trueWeights = new double[numFeatures];
        {
            Random rng = new Random(42);
            for (int i = 0; i < numFeatures; i++) trueWeights[i] = rng.nextGaussian();
        }

        // Generate labeled data: (features, label = dot(trueWeights, features) + noise)
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
                            double label = 0;
                            for (int i = 0; i < numFeatures; i++) label += trueWeights[i] * features[i];
                            label += rng.nextGaussian() * 0.1;
                            out.collect(new Tuple2<>(features, label));
                        }
                    }
                });

        // Initialise weights to zero
        double[] initWeights = new double[numFeatures];
        DataSet<double[]> weights = env.fromElements(initWeights);

        IterativeDataSet<double[]> loop = weights.iterate(maxIterations);

        // Compute gradient: grad = (1/n) * sum_i (pred_i - y_i) * x_i + reg * w
        DataSet<double[]> gradient = data
                .map(new ComputeGradient(numFeatures, regParam))
                .withBroadcastSet(loop, "weights")
                .reduce(new ReduceFunction<double[]>() {
                    @Override
                    public double[] reduce(double[] a, double[] b) {
                        double[] result = new double[a.length];
                        for (int i = 0; i < a.length; i++) result[i] = a[i] + b[i];
                        return result;
                    }
                });

        // Update weights: w = w - lr * grad
        DataSet<double[]> newWeights = gradient
                .map(new UpdateWeights(numFeatures, learningRate))
                .withBroadcastSet(loop, "weights");

        DataSet<double[]> finalWeights = loop.closeWith(newWeights);

        // Compute final RMSE
        DataSet<Double> squaredErrorSum = data
                .map(new ComputeSquaredError(numFeatures))
                .withBroadcastSet(finalWeights, "weights")
                .reduce(new ReduceFunction<Double>() {
                    @Override
                    public Double reduce(Double a, Double b) {
                        return a + b;
                    }
                });

        List<Double> sumList = squaredErrorSum.collect();
        double rmse = sumList.isEmpty() ? 0.0 : Math.sqrt(sumList.get(0) / numExamples);
        System.out.println("Linear Regression final RMSE: " + String.format("%.4f", rmse));
        writeResult(outputPath, "LinearRegression RMSE=" + String.format("%.4f", rmse) + "\n");

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

    public static class ComputeGradient extends RichMapFunction<Tuple2<double[], Double>, double[]> {

        private final int numFeatures;
        private final double regParam;
        private Collection<double[]> weightsCollection;

        public ComputeGradient(int numFeatures, double regParam) {
            this.numFeatures = numFeatures;
            this.regParam = regParam;
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
            double pred = 0;
            for (int i = 0; i < numFeatures; i++) pred += w[i] * x[i];
            double err = pred - y;
            double[] grad = new double[numFeatures + 1]; // last element = squared error for loss tracking
            for (int i = 0; i < numFeatures; i++) {
                grad[i] = err * x[i] + regParam * w[i];
            }
            grad[numFeatures] = err * err; // accumulate loss
            return grad;
        }
    }

    public static class UpdateWeights extends RichMapFunction<double[], double[]> {

        private final int numFeatures;
        private final double learningRate;
        private Collection<double[]> weightsCollection;

        public UpdateWeights(int numFeatures, double learningRate) {
            this.numFeatures = numFeatures;
            this.learningRate = learningRate;
        }

        @Override
        public void open(Configuration parameters) {
            weightsCollection = getRuntimeContext().getBroadcastVariable("weights");
        }

        @Override
        public double[] map(double[] gradSum) {
            double[] w = weightsCollection.iterator().next();
            double[] newW = new double[numFeatures];
            // gradSum is already sum; divide by count approximation (treat as per-sample)
            for (int i = 0; i < numFeatures; i++) {
                newW[i] = w[i] - learningRate * gradSum[i];
            }
            return newW;
        }
    }

    public static class ComputeSquaredError extends RichMapFunction<Tuple2<double[], Double>, Double> {

        private final int numFeatures;
        private Collection<double[]> weightsCollection;

        public ComputeSquaredError(int numFeatures) {
            this.numFeatures = numFeatures;
        }

        @Override
        public void open(Configuration parameters) {
            weightsCollection = getRuntimeContext().getBroadcastVariable("weights");
        }

        @Override
        public Double map(Tuple2<double[], Double> example) {
            double[] w = weightsCollection.iterator().next();
            double pred = 0;
            for (int i = 0; i < numFeatures; i++) pred += w[i] * example.f0[i];
            double err = pred - example.f1;
            return err * err;
        }
    }
}
