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

import org.apache.flink.api.common.functions.MapPartitionFunction;
import org.apache.flink.api.common.functions.ReduceFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.util.Collector;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.util.Random;

/**
 * Flink batch HDFS I/O benchmark (DFSIO equivalent).
 * Write phase: each parallel task writes a file of file_size_mb MB to HDFS.
 * Read phase: each parallel task reads a file from HDFS.
 * Measures and prints aggregate throughput.
 * Args: &lt;input_path&gt; &lt;output_path&gt; &lt;num_files&gt; &lt;file_size_mb&gt; &lt;read_only&gt;
 */
public class FlinkDFSIOE {

    private static final int BUFFER_SIZE = 1024 * 1024; // 1 MB buffer

    public static void main(String[] args) throws Exception {
        if (args.length < 5) {
            System.err.println(
                "Usage: FlinkDFSIOE <input_path> <output_path> <num_files> <file_size_mb> <read_only>");
            System.exit(1);
        }
        final String inputPath = args[0];
        final String outputPath = args[1];
        final int numFiles = Integer.parseInt(args[2]);
        final long fileSizeMb = Long.parseLong(args[3]);
        final boolean readOnly = Boolean.parseBoolean(args[4]);

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Create a dataset with one element per file
        DataSet<Integer> fileIds = env.generateSequence(0, numFiles - 1)
                .map(l -> l.intValue())
                .returns(org.apache.flink.api.common.typeinfo.BasicTypeInfo.INT_TYPE_INFO);

        // Write phase
        if (!readOnly) {
            DataSet<Tuple2<Long, Double>> writeStats = fileIds
                    .mapPartition(new MapPartitionFunction<Integer, Tuple2<Long, Double>>() {
                        @Override
                        public void mapPartition(Iterable<Integer> values,
                                Collector<Tuple2<Long, Double>> out) throws Exception {
                            Configuration hadoopConf = new Configuration();
                            for (Integer fileId : values) {
                                Path filePath = new Path(outputPath + "/part-" + fileId);
                                FileSystem fs = filePath.getFileSystem(hadoopConf);
                                byte[] buffer = new byte[BUFFER_SIZE];
                                new Random(fileId).nextBytes(buffer);
                                long totalBytes = fileSizeMb * 1024 * 1024;
                                long bytesWritten = 0;
                                long startTime = System.currentTimeMillis();
                                try (FSDataOutputStream out2 = fs.create(filePath, true)) {
                                    while (bytesWritten < totalBytes) {
                                        long remaining = totalBytes - bytesWritten;
                                        int toWrite = (int) Math.min(remaining, buffer.length);
                                        out2.write(buffer, 0, toWrite);
                                        bytesWritten += toWrite;
                                    }
                                }
                                double elapsed = (System.currentTimeMillis() - startTime) / 1000.0;
                                double throughputMBs = elapsed > 0
                                        ? (bytesWritten / 1024.0 / 1024.0) / elapsed : 0;
                                out.collect(new Tuple2<>(bytesWritten, throughputMBs));
                            }
                        }
                    });

            DataSet<Tuple2<Long, Double>> writeSummary = writeStats
                    .reduce(new ReduceFunction<Tuple2<Long, Double>>() {
                        @Override
                        public Tuple2<Long, Double> reduce(Tuple2<Long, Double> a,
                                Tuple2<Long, Double> b) {
                            return new Tuple2<>(a.f0 + b.f0, a.f1 + b.f1);
                        }
                    });

            java.util.List<Tuple2<Long, Double>> writeResult = writeSummary.collect();
            if (!writeResult.isEmpty()) {
                Tuple2<Long, Double> ws = writeResult.get(0);
                System.out.printf("Write phase: total bytes=%d, aggregate throughput=%.2f MB/s%n",
                        ws.f0, ws.f1);
            }
        }

        // Read phase
        String readPath = readOnly ? inputPath : outputPath;
        DataSet<Tuple2<Long, Double>> readStats = fileIds
                .mapPartition(new MapPartitionFunction<Integer, Tuple2<Long, Double>>() {
                    @Override
                    public void mapPartition(Iterable<Integer> values,
                            Collector<Tuple2<Long, Double>> out) throws Exception {
                        Configuration hadoopConf = new Configuration();
                        for (Integer fileId : values) {
                            Path filePath = new Path(readPath + "/part-" + fileId);
                            FileSystem fs = filePath.getFileSystem(hadoopConf);
                            if (!fs.exists(filePath)) {
                                out.collect(new Tuple2<>(0L, 0.0));
                                continue;
                            }
                            byte[] buffer = new byte[BUFFER_SIZE];
                            long bytesRead = 0;
                            long startTime = System.currentTimeMillis();
                            try (FSDataInputStream in = fs.open(filePath)) {
                                int read;
                                while ((read = in.read(buffer)) >= 0) {
                                    bytesRead += read;
                                }
                            }
                            double elapsed = (System.currentTimeMillis() - startTime) / 1000.0;
                            double throughputMBs = elapsed > 0
                                    ? (bytesRead / 1024.0 / 1024.0) / elapsed : 0;
                            out.collect(new Tuple2<>(bytesRead, throughputMBs));
                        }
                    }
                });

        DataSet<Tuple2<Long, Double>> readSummary = readStats
                .reduce(new ReduceFunction<Tuple2<Long, Double>>() {
                    @Override
                    public Tuple2<Long, Double> reduce(Tuple2<Long, Double> a,
                            Tuple2<Long, Double> b) {
                        return new Tuple2<>(a.f0 + b.f0, a.f1 + b.f1);
                    }
                });

        java.util.List<Tuple2<Long, Double>> readResult = readSummary.collect();
        if (!readResult.isEmpty()) {
            Tuple2<Long, Double> rs = readResult.get(0);
            System.out.printf("Read phase: total bytes=%d, aggregate throughput=%.2f MB/s%n",
                    rs.f0, rs.f1);
        }
        // collect() already triggers env.execute() internally — no extra execute() needed
    }
}
