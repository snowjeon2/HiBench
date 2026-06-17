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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.operators.IterativeDataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;

/**
 * Flink batch PageRank using the DataSet iterative API.
 * Input: edge list where each line is "srcId dstId" (space-separated long integers).
 *
 * Avoids distinct() and count() — instead initialises ranks from outDegree groupBy,
 * which only requires one sort pass per source vertex rather than a global distinct.
 */
public class FlinkPageRank {

    private static final double DAMPING = 0.85;

    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.err.println("Usage: FlinkPageRank <input_edges_path> <output_path> <num_iterations>");
            System.exit(1);
        }
        final String inputPath    = args[0];
        final String outputPath   = args[1];
        final int numIterations   = Integer.parseInt(args[2]);

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Read edges: "srcId dstId"
        DataSet<Tuple2<Long, Long>> edges = env.readTextFile(inputPath)
            .filter(line -> !line.isEmpty() && !line.startsWith("#"))
            .map(new MapFunction<String, Tuple2<Long, Long>>() {
                @Override
                public Tuple2<Long, Long> map(String line) throws Exception {
                    String[] p = line.trim().split("\\s+");
                    return new Tuple2<>(Long.parseLong(p[0]), Long.parseLong(p[1]));
                }
            });

        // Out-degree: (srcId, degree)  — groupBy replaces the expensive distinct()
        DataSet<Tuple2<Long, Long>> outDegree = edges
            .map(new MapFunction<Tuple2<Long, Long>, Tuple2<Long, Long>>() {
                @Override
                public Tuple2<Long, Long> map(Tuple2<Long, Long> e) {
                    return new Tuple2<>(e.f0, 1L);
                }
            })
            .groupBy(0)
            .sum(1);

        // Edge weights: (srcId, dstId, 1/outDegree)
        DataSet<Tuple3<Long, Long, Double>> edgesWithWeight = edges
            .join(outDegree).where(0).equalTo(0)
            .map(new MapFunction<Tuple2<Tuple2<Long, Long>, Tuple2<Long, Long>>,
                    Tuple3<Long, Long, Double>>() {
                @Override
                public Tuple3<Long, Long, Double> map(
                        Tuple2<Tuple2<Long, Long>, Tuple2<Long, Long>> j) {
                    return new Tuple3<>(j.f0.f0, j.f0.f1, 1.0 / j.f1.f1);
                }
            });

        // Initial ranks from source vertices only — no separate count() job needed.
        // Rank starts at 1.0 (standard normalisation-free initialisation).
        DataSet<Tuple2<Long, Double>> ranks = outDegree
            .map(new MapFunction<Tuple2<Long, Long>, Tuple2<Long, Double>>() {
                @Override
                public Tuple2<Long, Double> map(Tuple2<Long, Long> t) {
                    return new Tuple2<>(t.f0, 1.0);
                }
            });

        IterativeDataSet<Tuple2<Long, Double>> iteration = ranks.iterate(numIterations);

        // Contributions: for each edge (src->dst, w), dst gets current_rank(src) * w
        DataSet<Tuple2<Long, Double>> contributions = edgesWithWeight
            .join(iteration).where(0).equalTo(0)
            .map(new MapFunction<Tuple2<Tuple3<Long, Long, Double>, Tuple2<Long, Double>>,
                    Tuple2<Long, Double>>() {
                @Override
                public Tuple2<Long, Double> map(
                        Tuple2<Tuple3<Long, Long, Double>, Tuple2<Long, Double>> j) {
                    return new Tuple2<>(j.f0.f1, j.f1.f1 * j.f0.f2);
                }
            })
            .groupBy(0)
            .sum(1);

        // Apply damping: rank = (1 - d) + d * sum(contributions)
        DataSet<Tuple2<Long, Double>> newRanks = contributions
            .map(new MapFunction<Tuple2<Long, Double>, Tuple2<Long, Double>>() {
                @Override
                public Tuple2<Long, Double> map(Tuple2<Long, Double> c) {
                    return new Tuple2<>(c.f0, (1.0 - DAMPING) + DAMPING * c.f1);
                }
            });

        DataSet<Tuple2<Long, Double>> finalRanks = iteration.closeWith(newRanks);

        finalRanks.writeAsCsv(outputPath, "\n", " ");
        env.execute("Flink PageRank");
    }
}
