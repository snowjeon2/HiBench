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
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.util.Collector;

import java.util.Random;

/**
 * Flink batch summarizer benchmark.
 * Generates random dense vectors and computes per-feature min, max, mean, variance.
 * Args: &lt;input_path&gt; &lt;num_examples&gt; &lt;num_features&gt;
 */
public class FlinkSummarizer {

    /**
     * Accumulator for online computation of min, max, sum, sum-of-squares per feature.
     * Layout: [min_0..min_{d-1}, max_0..max_{d-1}, sum_0..sum_{d-1}, sumSq_0..sumSq_{d-1}, count]
     */
    public static class SummaryAcc implements java.io.Serializable {
        public double[] min;
        public double[] max;
        public double[] sum;
        public double[] sumSq;
        public long count;

        public SummaryAcc() {}

        public SummaryAcc(int numFeatures) {
            min = new double[numFeatures];
            max = new double[numFeatures];
            sum = new double[numFeatures];
            sumSq = new double[numFeatures];
            count = 0;
            java.util.Arrays.fill(min, Double.MAX_VALUE);
            java.util.Arrays.fill(max, -Double.MAX_VALUE);
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("count=").append(count).append("\n");
            for (int i = 0; i < Math.min(min.length, 5); i++) {
                sb.append("feature[").append(i).append("] min=").append(String.format("%.4f", min[i]))
                  .append(" max=").append(String.format("%.4f", max[i]))
                  .append(" mean=").append(String.format("%.4f", sum[i]/count))
                  .append("\n");
            }
            return sb.toString();
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println("Usage: FlinkSummarizer <input_path> <num_examples> <num_features> <output_path>");
            System.exit(1);
        }
        // input_path is accepted for API compatibility but data is generated in-memory
        final long numExamples = Long.parseLong(args[1]);
        final int numFeatures = Integer.parseInt(args[2]);
        final String outputPath = args[3];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Generate num_examples random dense vectors
        DataSet<double[]> vectors = env.generateSequence(0, numExamples - 1)
                .mapPartition(new MapPartitionFunction<Long, double[]>() {
                    @Override
                    public void mapPartition(Iterable<Long> values, Collector<double[]> out) {
                        Random rng = new Random();
                        for (Long ignored : values) {
                            double[] vec = new double[numFeatures];
                            for (int d = 0; d < numFeatures; d++) {
                                vec[d] = rng.nextGaussian();
                            }
                            out.collect(vec);
                        }
                    }
                });

        // Map each vector to a SummaryAcc initialised from that single point
        DataSet<SummaryAcc> perPoint = vectors.map(new MapFunction<double[], SummaryAcc>() {
            @Override
            public SummaryAcc map(double[] vec) {
                SummaryAcc acc = new SummaryAcc(vec.length);
                acc.count = 1;
                for (int d = 0; d < vec.length; d++) {
                    acc.min[d] = vec[d];
                    acc.max[d] = vec[d];
                    acc.sum[d] = vec[d];
                    acc.sumSq[d] = vec[d] * vec[d];
                }
                return acc;
            }
        });

        // Reduce to global summary
        DataSet<SummaryAcc> summary = perPoint.reduce(new ReduceFunction<SummaryAcc>() {
            @Override
            public SummaryAcc reduce(SummaryAcc a, SummaryAcc b) {
                SummaryAcc result = new SummaryAcc(a.min.length);
                result.count = a.count + b.count;
                for (int d = 0; d < a.min.length; d++) {
                    result.min[d] = Math.min(a.min[d], b.min[d]);
                    result.max[d] = Math.max(a.max[d], b.max[d]);
                    result.sum[d] = a.sum[d] + b.sum[d];
                    result.sumSq[d] = a.sumSq[d] + b.sumSq[d];
                }
                return result;
            }
        });

        // Write summary statistics to HDFS
        summary.writeAsText(outputPath);
        env.execute("Flink Summarizer");
    }
}
