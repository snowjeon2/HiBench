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

package com.intel.hibench.flinkbench.graph;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.operators.IterativeDataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.util.Collector;

/**
 * Flink batch NWeight graph benchmark.
 * Propagates vertex weights to N-hop neighbours using IterativeDataSet.
 * Input edge list format: "srcId dstId weight" or "srcId dstId" (space-separated).
 * Args: &lt;input_path&gt; &lt;output_path&gt; &lt;degree&gt; &lt;max_out_edges&gt;
 */
public class FlinkNWeight {

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println(
                "Usage: FlinkNWeight <input_path> <output_path> <degree> <max_out_edges>");
            System.exit(1);
        }
        final String inputPath = args[0];
        final String outputPath = args[1];
        final int degree = Integer.parseInt(args[2]);
        final int maxOutEdges = Integer.parseInt(args[3]);

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Read adjacency list: "<src>\t<dst1>:<w1>,<dst2>:<w2>,..."
        DataSet<Tuple3<Long, Long, Double>> edges = env.readTextFile(inputPath)
                .flatMap(new FlatMapFunction<String, Tuple3<Long, Long, Double>>() {
                    @Override
                    public void flatMap(String line, Collector<Tuple3<Long, Long, Double>> out) {
                        if (line.isEmpty() || line.startsWith("#")) return;
                        String[] fields = line.trim().split("\\s+", 2);
                        if (fields.length < 2) return;
                        long src;
                        try { src = Long.parseLong(fields[0].trim()); }
                        catch (NumberFormatException e) { return; }
                        for (String pairStr : fields[1].split("[,\\s]+")) {
                            if (pairStr.isEmpty()) continue;
                            String[] pair = pairStr.split(":");
                            if (pair.length < 2) continue;
                            try {
                                long dst = Long.parseLong(pair[0].trim());
                                double weight = Double.parseDouble(pair[1]);
                                out.collect(new Tuple3<>(src, dst, weight));
                            } catch (NumberFormatException e) { /* skip malformed */ }
                        }
                    }
                });

        // Cap max out-edges per vertex
        DataSet<Tuple3<Long, Long, Double>> cappedEdges = edges
                .groupBy(0)
                .sortGroup(2, org.apache.flink.api.common.operators.Order.DESCENDING)
                .first(maxOutEdges);

        // Collect all vertex IDs and initialise with weight = 1.0
        DataSet<Long> vertices = edges
                .flatMap(new FlatMapFunction<Tuple3<Long, Long, Double>, Long>() {
                    @Override
                    public void flatMap(Tuple3<Long, Long, Double> e, Collector<Long> out) {
                        out.collect(e.f0);
                        out.collect(e.f1);
                    }
                })
                .distinct();

        DataSet<Tuple2<Long, Double>> vertexWeights = vertices
                .map(new MapFunction<Long, Tuple2<Long, Double>>() {
                    @Override
                    public Tuple2<Long, Double> map(Long v) {
                        return new Tuple2<>(v, 1.0);
                    }
                });

        // Iterate for 'degree' hops: propagate weights along edges
        IterativeDataSet<Tuple2<Long, Double>> loop = vertexWeights.iterate(degree);

        // Propagate: for each edge (src, dst, edgeWeight), new weight for dst += src_weight * edgeWeight
        DataSet<Tuple2<Long, Double>> contributions = cappedEdges
                .join(loop).where(0).equalTo(0)
                .map(new MapFunction<Tuple2<Tuple3<Long, Long, Double>, Tuple2<Long, Double>>,
                        Tuple2<Long, Double>>() {
                    @Override
                    public Tuple2<Long, Double> map(
                            Tuple2<Tuple3<Long, Long, Double>, Tuple2<Long, Double>> joined) {
                        long dst = joined.f0.f1;
                        double edgeWeight = joined.f0.f2;
                        double srcWeight = joined.f1.f1;
                        return new Tuple2<>(dst, srcWeight * edgeWeight);
                    }
                })
                .groupBy(0)
                .reduce(new ReduceFunction<Tuple2<Long, Double>>() {
                    @Override
                    public Tuple2<Long, Double> reduce(Tuple2<Long, Double> a,
                            Tuple2<Long, Double> b) {
                        return new Tuple2<>(a.f0, a.f1 + b.f1);
                    }
                });

        // Merge: vertices that received contributions get new weight, others keep current
        DataSet<Tuple2<Long, Double>> newWeights = loop
                .leftOuterJoin(contributions).where(0).equalTo(0)
                .with(new org.apache.flink.api.common.functions.JoinFunction<
                        Tuple2<Long, Double>, Tuple2<Long, Double>, Tuple2<Long, Double>>() {
                    @Override
                    public Tuple2<Long, Double> join(Tuple2<Long, Double> current,
                            Tuple2<Long, Double> contrib) {
                        if (contrib == null) return current;
                        return new Tuple2<>(current.f0, contrib.f1);
                    }
                });

        DataSet<Tuple2<Long, Double>> finalWeights = loop.closeWith(newWeights);

        finalWeights.writeAsCsv(outputPath, "\n", " ");
        env.execute("Flink NWeight");
    }
}
