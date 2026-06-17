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

package com.intel.hibench.flinkbench.micro;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.util.Collector;

public class FlinkWordCount {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: FlinkWordCount <input_path> <output_path>");
            System.exit(1);
        }
        String inputPath = args[0];
        String outputPath = args[1];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        DataSet<String> text = env.readTextFile(inputPath);

        DataSet<Tuple2<String, Integer>> counts = text
            .flatMap(new FlatMapFunction<String, Tuple2<String, Integer>>() {
                @Override
                public void flatMap(String line, Collector<Tuple2<String, Integer>> out) {
                    for (String word : line.split("\\s+")) {
                        if (!word.isEmpty()) {
                            out.collect(new Tuple2<>(word, 1));
                        }
                    }
                }
            })
            .groupBy(0)
            .sum(1);

        counts.writeAsCsv(outputPath, "\n", " ");
        env.execute("WordCount");
    }
}
