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

import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.common.functions.GroupReduceFunction;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.operators.IterativeDataSet;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.api.java.tuple.Tuple3;
import org.apache.flink.util.Collector;

import java.util.List;
import java.util.Random;

/**
 * Flink batch LDA using IterativeDataSet (single Flink job — avoids per-iteration job submission
 * overhead). Uses join instead of withBroadcastSet to avoid broadcast-inside-iteration crash.
 * Args: &lt;num_topics&gt; &lt;num_iterations&gt; &lt;num_documents&gt; &lt;num_vocabulary&gt;
 */
public class FlinkLDA {

    private static final double BETA         = 0.01;
    private static final int    WORDS_PER_DOC = 50;

    public static void main(String[] args) throws Exception {
        if (args.length < 5) {
            System.err.println(
                "Usage: FlinkLDA <num_topics> <num_iterations> <num_documents> <num_vocabulary> <output_path>");
            System.exit(1);
        }
        final int numTopics = Integer.parseInt(args[0]);
        final int maxIter   = Integer.parseInt(args[1]);
        final int numDocs   = Integer.parseInt(args[2]);
        final int vocabSize = Integer.parseInt(args[3]);
        final String outputPath = args[4];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Corpus: (docId, wordId) — static across iterations
        DataSet<Tuple2<Integer, Integer>> corpus = env
                .generateSequence(0, (long) numDocs - 1)
                .mapPartition(new MapPartitionFunction<Long, Tuple2<Integer, Integer>>() {
                    @Override
                    public void mapPartition(Iterable<Long> values,
                            Collector<Tuple2<Integer, Integer>> out) {
                        for (Long docId : values) {
                            Random rng = new Random(docId);
                            int topic1 = (int)(docId % numTopics);
                            int topic2 = (topic1 + 1) % numTopics;
                            for (int w = 0; w < WORDS_PER_DOC; w++) {
                                int topic = rng.nextBoolean() ? topic1 : topic2;
                                int wordBase = topic * (vocabSize / Math.max(numTopics, 1));
                                int offset = Math.abs((int)(rng.nextGaussian() * 10));
                                int wordId = (wordBase + offset) % vocabSize;
                                out.collect(new Tuple2<>(docId.intValue(), wordId));
                            }
                        }
                    }
                });

        // Initial topic-word counts: (topicId, wordId, count)
        DataSet<Tuple3<Integer, Integer, Double>> initCounts = env
                .generateSequence(0, (long) numTopics * vocabSize - 1)
                .map(new MapFunction<Long, Tuple3<Integer, Integer, Double>>() {
                    @Override
                    public Tuple3<Integer, Integer, Double> map(Long idx) {
                        int t = (int)(idx / vocabSize);
                        int v = (int)(idx % vocabSize);
                        Random rng = new Random(idx);
                        return new Tuple3<>(t, v, rng.nextDouble() + BETA);
                    }
                });

        // IterativeDataSet: single Flink job, Flink handles iterations internally
        IterativeDataSet<Tuple3<Integer, Integer, Double>> loop = initCounts.iterate(maxIter);

        // --- Normalise: compute topic totals ---
        DataSet<Tuple2<Integer, Double>> topicTotals = loop
                .map(new MapFunction<Tuple3<Integer, Integer, Double>, Tuple2<Integer, Double>>() {
                    @Override
                    public Tuple2<Integer, Double> map(Tuple3<Integer, Integer, Double> t) {
                        return new Tuple2<>(t.f0, t.f2);
                    }
                })
                .groupBy(0)
                .reduce(new ReduceFunction<Tuple2<Integer, Double>>() {
                    @Override
                    public Tuple2<Integer, Double> reduce(Tuple2<Integer, Double> a,
                            Tuple2<Integer, Double> b) {
                        return new Tuple2<>(a.f0, a.f1 + b.f1);
                    }
                });

        // Topic-word probabilities: (topicId, wordId, prob)
        DataSet<Tuple3<Integer, Integer, Double>> topicWordProbs = loop
                .join(topicTotals).where(0).equalTo(0)
                .map(new MapFunction<
                        Tuple2<Tuple3<Integer, Integer, Double>, Tuple2<Integer, Double>>,
                        Tuple3<Integer, Integer, Double>>() {
                    @Override
                    public Tuple3<Integer, Integer, Double> map(
                            Tuple2<Tuple3<Integer, Integer, Double>, Tuple2<Integer, Double>> j) {
                        return new Tuple3<>(j.f0.f0, j.f0.f1,
                                j.f0.f2 / Math.max(j.f1.f1, 1e-12));
                    }
                });

        // E-step: join corpus with topic-word probs on wordId
        // Produces (docId, wordId, topicId, prob)
        DataSet<Tuple3<Integer, Integer, Double>> assignments = corpus
                .join(topicWordProbs).where(1).equalTo(1)
                .with(new org.apache.flink.api.common.functions.JoinFunction<
                        Tuple2<Integer, Integer>,
                        Tuple3<Integer, Integer, Double>,
                        Tuple3<Integer, Integer, Double>>() {
                    @Override
                    public Tuple3<Integer, Integer, Double> join(
                            Tuple2<Integer, Integer> docWord,
                            Tuple3<Integer, Integer, Double> twProb) {
                        // (docId, wordId, topicId encoded as double)
                        // We pack (docId, wordId) as key and topicId+prob as value
                        // Emit: (docId*vocabSize+wordId, topicId, prob) — groupBy for argmax
                        return new Tuple3<>(docWord.f0 * vocabSize + docWord.f1,
                                twProb.f0, twProb.f2);
                    }
                });

        // Pick best topic per (docId, wordId) pair
        DataSet<Tuple2<Integer, Integer>> bestTopics = assignments
                .groupBy(0)
                .reduceGroup(new GroupReduceFunction<
                        Tuple3<Integer, Integer, Double>, Tuple2<Integer, Integer>>() {
                    @Override
                    public void reduce(Iterable<Tuple3<Integer, Integer, Double>> group,
                            Collector<Tuple2<Integer, Integer>> out) {
                        int bestTopic = 0;
                        double bestProb = -1;
                        int wordEncoded = -1;
                        for (Tuple3<Integer, Integer, Double> t : group) {
                            wordEncoded = t.f0;
                            if (t.f2 > bestProb) { bestProb = t.f2; bestTopic = t.f1; }
                        }
                        if (wordEncoded >= 0) {
                            int wordId = wordEncoded % vocabSize;
                            out.collect(new Tuple2<>(bestTopic, wordId));
                        }
                    }
                });

        // M-step: count (topicId, wordId) assignments → new topic-word counts
        DataSet<Tuple3<Integer, Integer, Double>> newCounts = bestTopics
                .map(new MapFunction<Tuple2<Integer, Integer>, Tuple3<Integer, Integer, Double>>() {
                    @Override
                    public Tuple3<Integer, Integer, Double> map(Tuple2<Integer, Integer> t) {
                        return new Tuple3<>(t.f0, t.f1, 1.0);
                    }
                })
                .groupBy(0, 1)
                .reduce(new ReduceFunction<Tuple3<Integer, Integer, Double>>() {
                    @Override
                    public Tuple3<Integer, Integer, Double> reduce(
                            Tuple3<Integer, Integer, Double> a,
                            Tuple3<Integer, Integer, Double> b) {
                        return new Tuple3<>(a.f0, a.f1, a.f2 + b.f2 + BETA);
                    }
                });

        DataSet<Tuple3<Integer, Integer, Double>> result = loop.closeWith(newCounts);

        // Print a summary (triggers execution)
        List<Tuple3<Integer, Integer, Double>> top =
                FlinkJobUtils.collect(result.first(numTopics * 3), env, "LDA");
        System.out.println("LDA complete (" + maxIter + " iterations). Sample topic-word counts:");
        for (Tuple3<Integer, Integer, Double> t : top) {
            System.out.printf("  topic=%d word=%d count=%.2f%n", t.f0, t.f1, t.f2);
        }
        writeResult(outputPath, "LDA topics=" + numTopics + " iterations=" + maxIter + " docs=" + numDocs + "\n");
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
}
