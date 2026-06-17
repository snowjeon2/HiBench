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

import org.apache.flink.api.common.operators.Order;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.tuple.Tuple1;

public class FlinkSort {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: FlinkSort <input_path> <output_path>");
            System.exit(1);
        }
        String inputPath = args[0];
        String outputPath = args[1];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Wrap each line in Tuple1 to enable range partitioning and sort
        DataSet<Tuple1<String>> data = env.readTextFile(inputPath)
            .map(line -> new Tuple1<>(line))
            .returns(new org.apache.flink.api.java.typeutils.TupleTypeInfo<>(
                org.apache.flink.api.common.typeinfo.BasicTypeInfo.STRING_TYPE_INFO));

        DataSet<Tuple1<String>> sorted = data
            .partitionByHash(0)
            .sortPartition(0, Order.ASCENDING);

        sorted.map(t -> t.f0)
              .returns(org.apache.flink.api.common.typeinfo.BasicTypeInfo.STRING_TYPE_INFO)
              .writeAsText(outputPath);

        env.execute("Sort");
    }
}
