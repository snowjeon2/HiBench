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
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Flink batch Random Forest classifier.
 * Generates multi-class labeled data and builds an ensemble of decision trees
 * in parallel using mapPartition with bootstrap sampling.
 * Args: &lt;input_path&gt; &lt;num_examples&gt; &lt;num_features&gt; &lt;num_trees&gt; &lt;num_classes&gt; &lt;max_depth&gt;
 */
public class FlinkRandomForest {

    /** A simple binary decision tree node. */
    public static class TreeNode implements Serializable {
        public int featureIndex = -1;
        public double threshold = 0;
        public int label = -1; // leaf label (-1 if internal)
        public TreeNode left;
        public TreeNode right;

        public boolean isLeaf() {
            return label >= 0;
        }

        public int predict(double[] features) {
            if (isLeaf()) return label;
            if (features[featureIndex] <= threshold) {
                return left != null ? left.predict(features) : label;
            } else {
                return right != null ? right.predict(features) : label;
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 7) {
            System.err.println(
                "Usage: FlinkRandomForest <input_path> <num_examples> <num_features> "
                + "<num_trees> <num_classes> <max_depth> <output_path>");
            System.exit(1);
        }
        final long numExamples = Long.parseLong(args[1]);
        final int numFeatures = Integer.parseInt(args[2]);
        final int numTrees = Integer.parseInt(args[3]);
        final int numClasses = Integer.parseInt(args[4]);
        final int maxDepth = Integer.parseInt(args[5]);
        final String outputPath = args[6];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Generate multi-class data: label = argmax of random weights dot features
        DataSet<Tuple2<double[], Integer>> data = env.generateSequence(0, numExamples - 1)
                .mapPartition(new MapPartitionFunction<Long, Tuple2<double[], Integer>>() {
                    @Override
                    public void mapPartition(Iterable<Long> values,
                            Collector<Tuple2<double[], Integer>> out) {
                        Random rng = new Random();
                        for (Long seed : values) {
                            Random r2 = new Random(seed);
                            double[] features = new double[numFeatures];
                            for (int i = 0; i < numFeatures; i++) features[i] = r2.nextGaussian();
                            int label = (int) (Math.abs(seed) % numClasses);
                            out.collect(new Tuple2<>(features, label));
                        }
                    }
                });

        // Collect all data to driver for tree building
        final List<Tuple2<double[], Integer>> allData =
                FlinkJobUtils.collect(data, env, "RandomForest (load data)");

        // Build num_trees trees in parallel using generateSequence
        DataSet<TreeNode> trees = env.generateSequence(0, numTrees - 1)
                .mapPartition(new MapPartitionFunction<Long, TreeNode>() {
                    @Override
                    public void mapPartition(Iterable<Long> values, Collector<TreeNode> out) {
                        for (Long treeId : values) {
                            Random rng = new Random(treeId);
                            // Bootstrap sample
                            List<Tuple2<double[], Integer>> bootstrap = new ArrayList<>();
                            for (int i = 0; i < allData.size(); i++) {
                                bootstrap.add(allData.get(rng.nextInt(allData.size())));
                            }
                            TreeNode root = buildTree(bootstrap, numFeatures, numClasses, maxDepth, rng);
                            out.collect(root);
                        }
                    }

                    private TreeNode buildTree(List<Tuple2<double[], Integer>> samples,
                            int numFeat, int numCls, int depth, Random rng) {
                        if (samples.isEmpty() || depth == 0) {
                            return makeLeaf(samples, numCls);
                        }
                        // Count class distribution
                        int[] classCounts = new int[numCls];
                        for (Tuple2<double[], Integer> s : samples) classCounts[s.f1]++;
                        int majorityClass = 0;
                        for (int c = 1; c < numCls; c++) {
                            if (classCounts[c] > classCounts[majorityClass]) majorityClass = c;
                        }
                        // Check if pure
                        boolean pure = true;
                        for (int c = 0; c < numCls; c++) {
                            if (c != majorityClass && classCounts[c] > 0) { pure = false; break; }
                        }
                        if (pure) {
                            TreeNode leaf = new TreeNode();
                            leaf.label = majorityClass;
                            return leaf;
                        }
                        // Random feature subset (sqrt(numFeatures))
                        int subsetSize = Math.max(1, (int) Math.sqrt(numFeat));
                        int bestFeature = -1;
                        double bestThreshold = 0;
                        double bestGini = Double.MAX_VALUE;
                        for (int trial = 0; trial < subsetSize; trial++) {
                            int feat = rng.nextInt(numFeat);
                            // Pick median as threshold
                            double[] vals = new double[samples.size()];
                            for (int i = 0; i < samples.size(); i++) vals[i] = samples.get(i).f0[feat];
                            java.util.Arrays.sort(vals);
                            double threshold = vals[vals.length / 2];
                            double gini = computeGini(samples, feat, threshold, numCls);
                            if (gini < bestGini) {
                                bestGini = gini;
                                bestFeature = feat;
                                bestThreshold = threshold;
                            }
                        }
                        if (bestFeature < 0) {
                            return makeLeaf(samples, numCls);
                        }
                        List<Tuple2<double[], Integer>> leftSamples = new ArrayList<>();
                        List<Tuple2<double[], Integer>> rightSamples = new ArrayList<>();
                        for (Tuple2<double[], Integer> s : samples) {
                            if (s.f0[bestFeature] <= bestThreshold) leftSamples.add(s);
                            else rightSamples.add(s);
                        }
                        if (leftSamples.isEmpty() || rightSamples.isEmpty()) {
                            return makeLeaf(samples, numCls);
                        }
                        TreeNode node = new TreeNode();
                        node.featureIndex = bestFeature;
                        node.threshold = bestThreshold;
                        node.left = buildTree(leftSamples, numFeat, numCls, depth - 1, rng);
                        node.right = buildTree(rightSamples, numFeat, numCls, depth - 1, rng);
                        return node;
                    }

                    private TreeNode makeLeaf(List<Tuple2<double[], Integer>> samples, int numCls) {
                        int[] counts = new int[numCls];
                        for (Tuple2<double[], Integer> s : samples) counts[s.f1]++;
                        int best = 0;
                        for (int c = 1; c < numCls; c++) if (counts[c] > counts[best]) best = c;
                        TreeNode leaf = new TreeNode();
                        leaf.label = best;
                        return leaf;
                    }

                    private double computeGini(List<Tuple2<double[], Integer>> samples,
                            int feat, double threshold, int numCls) {
                        int[] leftCounts = new int[numCls];
                        int[] rightCounts = new int[numCls];
                        int leftTotal = 0, rightTotal = 0;
                        for (Tuple2<double[], Integer> s : samples) {
                            if (s.f0[feat] <= threshold) { leftCounts[s.f1]++; leftTotal++; }
                            else { rightCounts[s.f1]++; rightTotal++; }
                        }
                        double giniLeft = 1.0, giniRight = 1.0;
                        for (int c = 0; c < numCls; c++) {
                            if (leftTotal > 0) {
                                double p = (double) leftCounts[c] / leftTotal;
                                giniLeft -= p * p;
                            }
                            if (rightTotal > 0) {
                                double p = (double) rightCounts[c] / rightTotal;
                                giniRight -= p * p;
                            }
                        }
                        int total = leftTotal + rightTotal;
                        return (leftTotal * giniLeft + rightTotal * giniRight) / Math.max(total, 1);
                    }
                });

        // Evaluate ensemble accuracy using broadcast of all trees
        DataSet<double[]> accuracy = data
                .map(new EnsemblePredict(numClasses))
                .withBroadcastSet(trees, "trees")
                .reduce(new ReduceFunction<double[]>() {
                    @Override
                    public double[] reduce(double[] a, double[] b) {
                        return new double[]{a[0] + b[0], a[1] + b[1]};
                    }
                });

        List<double[]> accList = FlinkJobUtils.collect(accuracy, env, "RandomForest");
        double acc = accList.isEmpty() ? 0.0 : accList.get(0)[0] / accList.get(0)[1];
        System.out.println("Random Forest ensemble accuracy: " + String.format("%.4f", acc));
        writeResult(outputPath, "RandomForest accuracy=" + String.format("%.4f", acc) + "\n");

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

    public static class EnsemblePredict extends RichMapFunction<Tuple2<double[], Integer>, double[]> {

        private final int numClasses;
        private Collection<TreeNode> trees;

        public EnsemblePredict(int numClasses) {
            this.numClasses = numClasses;
        }

        @Override
        public void open(Configuration parameters) {
            trees = getRuntimeContext().getBroadcastVariable("trees");
        }

        @Override
        public double[] map(Tuple2<double[], Integer> example) {
            int[] votes = new int[numClasses];
            for (TreeNode tree : trees) {
                int pred = tree.predict(example.f0);
                if (pred >= 0 && pred < numClasses) votes[pred]++;
            }
            int bestLabel = 0;
            for (int c = 1; c < numClasses; c++) {
                if (votes[c] > votes[bestLabel]) bestLabel = c;
            }
            double correct = bestLabel == example.f1 ? 1.0 : 0.0;
            return new double[]{correct, 1.0};
        }
    }
}
