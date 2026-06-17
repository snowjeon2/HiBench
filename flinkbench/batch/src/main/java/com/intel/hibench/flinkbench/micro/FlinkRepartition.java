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

import com.intel.hibench.flinkbench.micro.TeraUtils.TeraRecordInputFormat;
import com.intel.hibench.flinkbench.micro.TeraUtils.TeraRecordOutputFormat;
import org.apache.flink.api.common.functions.MapFunction;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple2;

import java.util.Random;

public class FlinkRepartition {

    public static void main(String[] args) throws Exception {
        if (args.length < 4) {
            System.err.println(
                "Usage: FlinkRepartition <input_path> <output_path> <cache_in_memory> <disable_output>");
            System.exit(1);
        }
        String inputPath      = args[0];
        String outputPath     = args[1];
        boolean disableOutput = Boolean.parseBoolean(args[3]);

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();
        final int parallelism = env.getParallelism();

        TeraRecordInputFormat inputFormat = new TeraRecordInputFormat();
        inputFormat.setFilePath(inputPath);
        DataSet<byte[]> input = env.createInput(inputFormat);

        // Shuffle: assign random key → hash-partition → drop key
        DataSet<byte[]> repartitioned = input
            .map(new MapFunction<byte[], Tuple2<Integer, byte[]>>() {
                private final Random rng = new Random();
                @Override
                public Tuple2<Integer, byte[]> map(byte[] record) {
                    return new Tuple2<>(rng.nextInt(parallelism), record);
                }
            })
            .partitionByHash(0)
            .map(new MapFunction<Tuple2<Integer, byte[]>, byte[]>() {
                @Override
                public byte[] map(Tuple2<Integer, byte[]> t) { return t.f1; }
            });

        if (!disableOutput) {
            repartitioned.output(new TeraRecordOutputFormat(outputPath));
        } else {
            repartitioned.output(new org.apache.flink.api.java.io.DiscardingOutputFormat<>());
        }

        env.execute("Flink Repartition");
    }
}
