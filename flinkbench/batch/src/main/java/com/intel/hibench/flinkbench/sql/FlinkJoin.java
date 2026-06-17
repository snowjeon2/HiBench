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

package com.intel.hibench.flinkbench.sql;

import org.apache.flink.api.common.functions.JoinFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;

/**
 * Flink batch SQL Join benchmark.
 * Reads Rankings and UserVisits CSVs and computes:
 *   SELECT sourceIP, AVG(pageRank), SUM(adRevenue)
 *   FROM rankings JOIN uservisits ON pageURL=destURL
 *   WHERE visitDate BETWEEN '1999-01-01' AND '2000-01-01'
 *   GROUP BY sourceIP
 *
 * Rankings CSV format: pageURL,pageRank,avgDuration
 * UserVisits CSV format: sourceIP,destURL,visitDate,adRevenue,userAgent,...
 *
 * Args: &lt;input_path&gt; &lt;output_path&gt;
 *   input_path should have sub-directories: input_path/rankings and input_path/uservisits
 */
public class FlinkJoin {

    private static final String DATE_FROM = "1999-01-01";
    private static final String DATE_TO   = "2000-01-01";

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: FlinkJoin <input_path> <output_path>");
            System.exit(1);
        }
        final String inputPath = args[0];
        final String outputPath = args[1];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Read Rankings: (pageURL, pageRank)
        DataSet<Tuple2<String, Double>> rankings = env
                .readTextFile(inputPath + "/rankings")
                .filter(line -> !line.isEmpty() && !line.startsWith("#"))
                .map(new MapFunction<String, Tuple2<String, Double>>() {
                    @Override
                    public Tuple2<String, Double> map(String line) {
                        String[] fields = line.split(",", -1);
                        String pageURL = fields[0].trim();
                        double pageRank = 0.0;
                        try {
                            pageRank = Double.parseDouble(fields[1].trim());
                        } catch (NumberFormatException | ArrayIndexOutOfBoundsException e) {
                            pageRank = 0.0;
                        }
                        return new Tuple2<>(pageURL, pageRank);
                    }
                });

        // Read UserVisits filtered by date range: (destURL, sourceIP, adRevenue)
        DataSet<Tuple3<String, String, Double>> userVisits = env
                .readTextFile(inputPath + "/uservisits")
                .filter(line -> !line.isEmpty() && !line.startsWith("#"))
                .map(new MapFunction<String, Tuple3<String, String, Double>>() {
                    @Override
                    public Tuple3<String, String, Double> map(String line) {
                        String[] fields = line.split(",", -1);
                        String sourceIP = fields.length > 0 ? fields[0].trim() : "";
                        String destURL  = fields.length > 1 ? fields[1].trim() : "";
                        String visitDate = fields.length > 2 ? fields[2].trim() : "";
                        double adRevenue = 0.0;
                        try {
                            adRevenue = fields.length > 3 ? Double.parseDouble(fields[3].trim()) : 0.0;
                        } catch (NumberFormatException e) {
                            adRevenue = 0.0;
                        }
                        // Return null-sentinel if date out of range; filter below
                        if (visitDate.compareTo(DATE_FROM) >= 0 && visitDate.compareTo(DATE_TO) <= 0) {
                            return new Tuple3<>(destURL, sourceIP, adRevenue);
                        }
                        return new Tuple3<>("__FILTERED__", "", 0.0);
                    }
                })
                .filter(t -> !t.f0.equals("__FILTERED__"));

        // Join on pageURL == destURL: emit (sourceIP, pageRank, adRevenue)
        DataSet<Tuple3<String, Double, Double>> joined = rankings
                .join(userVisits).where(0).equalTo(0)
                .with(new JoinFunction<Tuple2<String, Double>, Tuple3<String, String, Double>,
                        Tuple3<String, Double, Double>>() {
                    @Override
                    public Tuple3<String, Double, Double> join(
                            Tuple2<String, Double> ranking,
                            Tuple3<String, String, Double> visit) {
                        // (sourceIP, pageRank, adRevenue)
                        return new Tuple3<>(visit.f1, ranking.f1, visit.f2);
                    }
                });

        // Group by sourceIP, aggregate AVG(pageRank) and SUM(adRevenue)
        // Use (sourceIP, sumPageRank, sumAdRevenue, count) for AVG computation
        DataSet<Tuple3<String, Double, Double>> aggregated = joined
                .map(new MapFunction<Tuple3<String, Double, Double>,
                        Tuple3<String, double[], Double>>() {
                    @Override
                    public Tuple3<String, double[], Double> map(Tuple3<String, Double, Double> t) {
                        return new Tuple3<>(t.f0, new double[]{t.f1, 1.0}, t.f2);
                    }
                })
                .groupBy(0)
                .reduce(new org.apache.flink.api.common.functions.ReduceFunction<
                        Tuple3<String, double[], Double>>() {
                    @Override
                    public Tuple3<String, double[], Double> reduce(
                            Tuple3<String, double[], Double> a,
                            Tuple3<String, double[], Double> b) {
                        return new Tuple3<>(a.f0,
                                new double[]{a.f1[0] + b.f1[0], a.f1[1] + b.f1[1]},
                                a.f2 + b.f2);
                    }
                })
                .map(new MapFunction<Tuple3<String, double[], Double>,
                        Tuple3<String, Double, Double>>() {
                    @Override
                    public Tuple3<String, Double, Double> map(Tuple3<String, double[], Double> t) {
                        double avgPageRank = t.f1[0] / Math.max(t.f1[1], 1.0);
                        return new Tuple3<>(t.f0, avgPageRank, t.f2);
                    }
                });

        aggregated.writeAsCsv(outputPath, "\n", ",");
        env.execute("Flink SQL Join");
    }
}
