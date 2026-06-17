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

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Random;

/**
 * Flink batch Gradient Boosted Trees (GBT) for regression/classification.
 * Uses simple regression stump (depth-1 trees) as weak learners.
 * Iterates manually for num_iterations boosting rounds.
 * Args: &lt;input_path&gt; &lt;num_examples&gt; &lt;num_features&gt; &lt;num_classes&gt;
 *       &lt;num_iterations&gt; &lt;max_depth&gt; &lt;learning_rate&gt;
 */
public class FlinkGBT {

    /** Simple regression tree for boosting. */
    public static class RegressionStump implements Serializable {
        public int featureIndex;
        public double threshold;
        public double leftValue;
        public double rightValue;

        public double predict(double[] features) {
            return features[featureIndex] <= threshold ? leftValue : rightValue;
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 8) {
            System.err.println(
                "Usage: FlinkGBT <input_path> <num_examples> <num_features> <num_classes> "
                + "<num_iterations> <max_depth> <learning_rate> <output_path>");
            System.exit(1);
        }
        final long numExamples = Long.parseLong(args[1]);
        final int numFeatures = Integer.parseInt(args[2]);
        final int numClasses = Integer.parseInt(args[3]);
        final int numIterations = Integer.parseInt(args[4]);
        final int maxDepth = Integer.parseInt(args[5]);
        final double learningRate = Double.parseDouble(args[6]);
        final String outputPath = args[7];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Generate labeled data: label = (hash of features) % numClasses
        DataSet<Tuple2<double[], Integer>> data = env.generateSequence(0, numExamples - 1)
                .mapPartition(new MapPartitionFunction<Long, Tuple2<double[], Integer>>() {
                    @Override
                    public void mapPartition(Iterable<Long> values,
                            Collector<Tuple2<double[], Integer>> out) {
                        for (Long seed : values) {
                            Random r = new Random(seed);
                            double[] features = new double[numFeatures];
                            for (int i = 0; i < numFeatures; i++) features[i] = r.nextGaussian();
                            int label = (int) (Math.abs(seed) % numClasses);
                            out.collect(new Tuple2<>(features, label));
                        }
                    }
                });

        final List<Tuple2<double[], Integer>> allData = data.collect();
        int n = allData.size();

        // Initialise predictions with base value (mean label for regression, 0.5 for binary)
        double[] predictions = new double[n];
        double meanLabel = 0;
        for (Tuple2<double[], Integer> t : allData) meanLabel += t.f1;
        meanLabel /= Math.max(n, 1);
        for (int i = 0; i < n; i++) predictions[i] = meanLabel;

        List<RegressionStump> ensemble = new ArrayList<>();

        for (int iter = 0; iter < numIterations; iter++) {
            // Compute negative gradients (pseudo-residuals) for MSE loss
            double[] residuals = new double[n];
            for (int i = 0; i < n; i++) {
                residuals[i] = allData.get(i).f1 - predictions[i];
            }

            // Fit a regression stump to the residuals
            RegressionStump stump = fitStump(allData, residuals, numFeatures, new Random(iter));
            ensemble.add(stump);

            // Update predictions
            for (int i = 0; i < n; i++) {
                predictions[i] += learningRate * stump.predict(allData.get(i).f0);
            }
        }

        // Compute final MSE
        double mse = 0;
        for (int i = 0; i < n; i++) {
            double err = allData.get(i).f1 - predictions[i];
            mse += err * err;
        }
        mse /= Math.max(n, 1);
        System.out.println("GBT final MSE loss after " + numIterations + " iterations: "
                + String.format("%.4f", mse));
        System.out.println("GBT final RMSE: " + String.format("%.4f", Math.sqrt(mse)));
        writeResult(outputPath, "GBT RMSE=" + String.format("%.4f", Math.sqrt(mse)) + "\n");

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

    private static RegressionStump fitStump(List<Tuple2<double[], Integer>> data,
            double[] residuals, int numFeatures, Random rng) {
        int n = data.size();
        RegressionStump bestStump = new RegressionStump();
        double bestMse = Double.MAX_VALUE;

        for (int feat = 0; feat < numFeatures; feat++) {
            // Use median as threshold
            double[] vals = new double[n];
            for (int i = 0; i < n; i++) vals[i] = data.get(i).f0[feat];
            java.util.Arrays.sort(vals);
            double threshold = vals[n / 2];

            double leftSum = 0, rightSum = 0;
            int leftCount = 0, rightCount = 0;
            for (int i = 0; i < n; i++) {
                if (data.get(i).f0[feat] <= threshold) {
                    leftSum += residuals[i];
                    leftCount++;
                } else {
                    rightSum += residuals[i];
                    rightCount++;
                }
            }
            double leftVal = leftCount > 0 ? leftSum / leftCount : 0;
            double rightVal = rightCount > 0 ? rightSum / rightCount : 0;

            double mse = 0;
            for (int i = 0; i < n; i++) {
                double pred = data.get(i).f0[feat] <= threshold ? leftVal : rightVal;
                double err = residuals[i] - pred;
                mse += err * err;
            }
            if (mse < bestMse) {
                bestMse = mse;
                bestStump.featureIndex = feat;
                bestStump.threshold = threshold;
                bestStump.leftValue = leftVal;
                bestStump.rightValue = rightVal;
            }
        }
        return bestStump;
    }
}
