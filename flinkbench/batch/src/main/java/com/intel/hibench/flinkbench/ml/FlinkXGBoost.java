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
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.util.Collector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Flink batch XGBoost-style gradient boosting using second-order Newton steps.
 * Uses Taylor expansion of the loss: g = first-order grad, h = second-order hessian.
 * Leaf weight = -sum(g) / (sum(h) + lambda).
 * Args: &lt;input_path&gt; &lt;num_examples&gt; &lt;num_features&gt; &lt;num_classes&gt;
 *       &lt;num_iterations&gt; &lt;max_depth&gt; &lt;learning_rate&gt;
 */
public class FlinkXGBoost {

    /** XGBoost-style regression stump with Newton leaf weights. */
    public static class XGBStump implements Serializable {
        public int featureIndex;
        public double threshold;
        public double leftWeight;
        public double rightWeight;

        public double predict(double[] features) {
            return features[featureIndex] <= threshold ? leftWeight : rightWeight;
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 8) {
            System.err.println(
                "Usage: FlinkXGBoost <input_path> <num_examples> <num_features> <num_classes> "
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
        final double regLambda = 1.0; // L2 regularisation on leaf weights

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Generate labeled data
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

        // Initialise predictions with mean label
        double[] predictions = new double[n];
        double meanLabel = 0;
        for (Tuple2<double[], Integer> t : allData) meanLabel += t.f1;
        meanLabel /= Math.max(n, 1);
        for (int i = 0; i < n; i++) predictions[i] = meanLabel;

        List<XGBStump> ensemble = new ArrayList<>();

        for (int iter = 0; iter < numIterations; iter++) {
            // Compute first-order gradient g and second-order hessian h for MSE loss
            // MSE: L = 0.5*(y - f)^2 => g = f - y, h = 1
            double[] g = new double[n];
            double[] h = new double[n];
            for (int i = 0; i < n; i++) {
                g[i] = predictions[i] - allData.get(i).f1;
                h[i] = 1.0;
            }

            // Fit XGBoost stump: leaf weight = -sum(g) / (sum(h) + lambda)
            XGBStump stump = fitXGBStump(allData, g, h, numFeatures, regLambda, new Random(iter));
            ensemble.add(stump);

            // Update predictions
            for (int i = 0; i < n; i++) {
                predictions[i] += learningRate * stump.predict(allData.get(i).f0);
            }
        }

        // Final loss
        double mse = 0;
        for (int i = 0; i < n; i++) {
            double err = allData.get(i).f1 - predictions[i];
            mse += err * err;
        }
        mse /= Math.max(n, 1);
        System.out.println("XGBoost final MSE after " + numIterations + " iterations: "
                + String.format("%.4f", mse));
        System.out.println("XGBoost final RMSE: " + String.format("%.4f", Math.sqrt(mse)));
        writeResult(outputPath, "XGBoost RMSE=" + String.format("%.4f", Math.sqrt(mse)) + "\n");

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

    private static XGBStump fitXGBStump(
            List<Tuple2<double[], Integer>> data,
            double[] g, double[] h,
            int numFeatures, double regLambda,
            Random rng) {
        int n = data.size();
        XGBStump bestStump = new XGBStump();
        double bestGain = Double.NEGATIVE_INFINITY;

        // Total G and H
        double totalG = 0, totalH = 0;
        for (int i = 0; i < n; i++) { totalG += g[i]; totalH += h[i]; }
        double baseScore = -(totalG * totalG) / (totalH + regLambda);

        for (int feat = 0; feat < numFeatures; feat++) {
            double[] vals = new double[n];
            for (int i = 0; i < n; i++) vals[i] = data.get(i).f0[feat];
            double[] sorted = vals.clone();
            java.util.Arrays.sort(sorted);
            double threshold = sorted[n / 2];

            double leftG = 0, leftH = 0;
            for (int i = 0; i < n; i++) {
                if (vals[i] <= threshold) { leftG += g[i]; leftH += h[i]; }
            }
            double rightG = totalG - leftG;
            double rightH = totalH - leftH;

            // Gain = 0.5 * [GL^2/(HL+lambda) + GR^2/(HR+lambda) - (GL+GR)^2/(HL+HR+lambda)]
            double gain = 0.5 * (
                (leftG * leftG) / (leftH + regLambda)
                + (rightG * rightG) / (rightH + regLambda)
                - (totalG * totalG) / (totalH + regLambda));

            if (gain > bestGain) {
                bestGain = gain;
                bestStump.featureIndex = feat;
                bestStump.threshold = threshold;
                // Newton leaf weights
                bestStump.leftWeight = -leftG / (leftH + regLambda);
                bestStump.rightWeight = -rightG / (rightH + regLambda);
            }
        }
        return bestStump;
    }
}
