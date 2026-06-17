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

package com.intel.hibench.flinkbench.websearch;

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.GroupReduceFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.util.Collector;

import java.util.ArrayList;
import java.util.List;

/**
 * Flink batch NutchIndexing benchmark.
 *
 * Reads web page text content from HDFS (one document per line, format: "url\tcontent"),
 * builds an inverted index mapping each word to the list of URLs containing it,
 * and writes the resulting index to the output path.
 *
 * This replicates the essential computation of Apache Nutch indexing:
 * tokenize page content and group postings by term.
 */
public class FlinkNutchIndexing {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: FlinkNutchIndexing <input_path> <output_path>");
            System.exit(1);
        }
        String inputPath = args[0];
        String outputPath = args[1];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Each line: "url\tcontent"  (Nutch crawldb segment format simplified to text)
        DataSet<String> pages = env.readTextFile(inputPath);

        // Tokenize: emit (word, url, 1) for each token in each page
        DataSet<Tuple2<String, String>> postings = pages
            .flatMap(new FlatMapFunction<String, Tuple2<String, String>>() {
                @Override
                public void flatMap(String line, Collector<Tuple2<String, String>> out) {
                    if (line.isEmpty()) return;
                    int tabIdx = line.indexOf('\t');
                    String url;
                    String content;
                    if (tabIdx >= 0) {
                        url = line.substring(0, tabIdx).trim();
                        content = line.substring(tabIdx + 1).trim();
                    } else {
                        url = line.trim();
                        content = line.trim();
                    }
                    for (String token : content.split("[^a-zA-Z0-9]+")) {
                        if (!token.isEmpty()) {
                            out.collect(new Tuple2<>(token.toLowerCase(), url));
                        }
                    }
                }
            });

        // Build inverted index: group by word, collect posting list
        DataSet<Tuple2<String, String>> invertedIndex = postings
            .groupBy(0)
            .reduceGroup(new GroupReduceFunction<Tuple2<String, String>, Tuple2<String, String>>() {
                @Override
                public void reduce(Iterable<Tuple2<String, String>> values,
                                   Collector<Tuple2<String, String>> out) {
                    String word = null;
                    List<String> urls = new ArrayList<>();
                    for (Tuple2<String, String> t : values) {
                        word = t.f0;
                        urls.add(t.f1);
                    }
                    if (word != null) {
                        out.collect(new Tuple2<>(word, String.join(",", urls)));
                    }
                }
            });

        invertedIndex.writeAsCsv(outputPath, "\n", "\t");
        env.execute("Flink NutchIndexing");
    }
}
