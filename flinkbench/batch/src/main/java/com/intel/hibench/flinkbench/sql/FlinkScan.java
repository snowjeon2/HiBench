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

import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;

/**
 * Flink batch SQL Scan benchmark.
 * Equivalent to: INSERT INTO uservisits_copy SELECT * FROM uservisits (full table scan).
 * Reads all UserVisits records from HDFS and writes them to the output path.
 * UserVisits CSV format: sourceIP,destURL,visitDate,adRevenue,userAgent,countryCode,
 *                         languageCode,searchWord,duration
 * Args: &lt;input_path&gt; &lt;output_path&gt;
 */
public class FlinkScan {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: FlinkScan <input_path> <output_path>");
            System.exit(1);
        }
        final String inputPath = args[0];
        final String outputPath = args[1];

        ExecutionEnvironment env = ExecutionEnvironment.getExecutionEnvironment();

        // Full table scan: read all records and write them out unchanged
        DataSet<String> records = env.readTextFile(inputPath)
                .filter(line -> !line.isEmpty() && !line.startsWith("#"));

        records.writeAsText(outputPath);
        env.execute("Scan");
    }
}
