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
import org.apache.flink.api.common.operators.Order;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.functions.KeySelector;

public class FlinkTeraSort {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: FlinkTeraSort <input_path> <output_path>");
            System.exit(1);
        }
        String inputPath  = args[0];
        String outputPath = args[1];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        TeraRecordInputFormat inputFormat = new TeraRecordInputFormat();
        inputFormat.setFilePath(inputPath);
        DataSet<byte[]> input = env.createInput(inputFormat);

        // Extract 10-byte key as hex string for range partitioning + sort
        KeySelector<byte[], String> keySelector = new KeySelector<byte[], String>() {
            @Override
            public String getKey(byte[] record) {
                return TeraUtils.keyHex(record);
            }
        };

        // Global sort: range-partition by key, then sort within each partition
        DataSet<byte[]> sorted = input
            .partitionByRange(keySelector)
            .sortPartition(keySelector, Order.ASCENDING);

        sorted.output(new TeraRecordOutputFormat(outputPath));

        env.execute("Flink TeraSort");
    }
}
