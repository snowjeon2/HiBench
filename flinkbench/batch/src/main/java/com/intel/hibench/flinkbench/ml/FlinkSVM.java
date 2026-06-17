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
 * Flink batch SVM with SGD (hinge loss) using IterativeDataSet.
 * Generates binary labeled data with labels in {-1, +1}.
 * Args: &lt;input_path&gt; &lt;num_examples&gt; &lt;num_features&gt; &lt;num_iterations&gt; &lt;step_size&gt;
 */
public class FlinkSVM {

    public static void main(String[] args) throws Exception {
        if (args.length < 6) {
            System.err.println(
                "Usage: FlinkSVM <input_path> <num_examples> <num_features> "
                + "<num_iterations> <step_size> <output_path>");
            System.exit(1);
        }
        final long numExamples = Long.parseLong(args[1]);
        final int numFeatures = Integer.parseInt(args[2]);
        final int numIterations = Integer.parseInt(args[3]);
        final double stepSize = Double.parseDouble(args[4]);
        final String outputPath = args[5];
        final double lambda = 0.01; // regularisation

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // True separating hyperplane weights
        final double[] trueWeights = new double[numFeatures];
        {
            Random rng = new Random(13);
            for (int i = 0; i < numFeatures; i++) trueWeights[i] = rng.nextGaussian();
        }

        // Generate data: label = sign(dot(w,x) + noise) in {-1, +1}
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
                            double margin = 0;
                            for (int i = 0; i < numFeatures; i++) margin += trueWeights[i] * features[i];
                            margin += rng.nextGaussian() * 0.5;
                            double label = margin >= 0 ? 1.0 : -1.0;
                            out.collect(new Tuple2<>(features, label));
                        }
                    }
                });

        // Initialise weights
        DataSet<double[]> weights = env.fromElements(new double[numFeatures]);

        IterativeDataSet<double[]> loop = weights.iterate(numIterations);

        // Compute subgradient of hinge loss + L2 regularisation
        // grad = lambda * w - (1/n) * sum_{margin < 1} y_i * x_i
        DataSet<double[]> gradSum = data
                .map(new SVMGradient(numFeatures, lambda))
                .withBroadcastSet(loop, "weights")
                .reduce(new ReduceFunction<double[]>() {
                    @Override
                    public double[] reduce(double[] a, double[] b) {
                        double[] result = new double[a.length];
                        for (int i = 0; i < a.length; i++) result[i] = a[i] + b[i];
                        return result;
                    }
                });

        DataSet<double[]> newWeights = gradSum
                .map(new RichMapFunction<double[], double[]>() {
                    private Collection<double[]> wColl;

                    @Override
                    public void open(Configuration parameters) {
                        wColl = getRuntimeContext().getBroadcastVariable("weights");
                    }

                    @Override
                    public double[] map(double[] gradSumArr) {
                        double[] w = wColl.iterator().next();
                        double[] newW = new double[numFeatures];
                        double scale = stepSize / numExamples;
                        for (int i = 0; i < numFeatures; i++) {
                            newW[i] = w[i] - scale * gradSumArr[i];
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
                        double score = 0;
                        for (int i = 0; i < numFeatures; i++) score += w[i] * example.f0[i];
                        double pred = score >= 0 ? 1.0 : -1.0;
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

        List<double[]> accList = accuracy.collect();
        double acc = accList.isEmpty() ? 0.0 : accList.get(0)[0] / accList.get(0)[1];
        System.out.println("SVM training accuracy: " + String.format("%.4f", acc));
        writeResult(outputPath, "SVM accuracy=" + String.format("%.4f", acc) + "\n");

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

    public static class SVMGradient extends RichMapFunction<Tuple2<double[], Double>, double[]> {

        private final int numFeatures;
        private final double lambda;
        private Collection<double[]> weightsCollection;

        public SVMGradient(int numFeatures, double lambda) {
            this.numFeatures = numFeatures;
            this.lambda = lambda;
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
            double margin = 0;
            for (int i = 0; i < numFeatures; i++) margin += w[i] * x[i];
            double[] grad = new double[numFeatures];
            // L2 regularisation gradient
            for (int i = 0; i < numFeatures; i++) grad[i] = lambda * w[i];
            // Hinge loss subgradient
            if (y * margin < 1.0) {
                for (int i = 0; i < numFeatures; i++) grad[i] -= y * x[i];
            }
            return grad;
        }
    }
}
