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

import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;

/**
 * Flink batch SQL Aggregation benchmark.
 * Reads UserVisits CSV and computes: SELECT sourceIP, SUM(adRevenue) GROUP BY sourceIP.
 * UserVisits CSV format: sourceIP,destURL,visitDate,adRevenue,userAgent,countryCode,
 *                         languageCode,searchWord,duration
 * Args: &lt;input_path&gt; &lt;output_path&gt;
 */
public class FlinkAggregation {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: FlinkAggregation <input_path> <output_path>");
            System.exit(1);
        }
        final String inputPath = args[0];
        final String outputPath = args[1];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Read UserVisits CSV: sourceIP,destURL,visitDate,adRevenue,...
        DataSet<Tuple2<String, Double>> sourceRevenue = env.readTextFile(inputPath)
                .filter(line -> !line.isEmpty() && !line.startsWith("#"))
                .map(new MapFunction<String, Tuple2<String, Double>>() {
                    @Override
                    public Tuple2<String, Double> map(String line) {
                        String[] fields = line.split(",", -1);
                        String sourceIP = fields[0].trim();
                        double adRevenue = 0.0;
                        try {
                            adRevenue = Double.parseDouble(fields[3].trim());
                        } catch (NumberFormatException | ArrayIndexOutOfBoundsException e) {
                            adRevenue = 0.0;
                        }
                        return new Tuple2<>(sourceIP, adRevenue);
                    }
                })
                .groupBy(0)
                .sum(1);

        sourceRevenue.writeAsCsv(outputPath, "\n", ",");
        env.execute("Aggregation");
    }
}
