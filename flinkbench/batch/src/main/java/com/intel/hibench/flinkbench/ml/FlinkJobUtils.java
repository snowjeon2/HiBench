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

import org.apache.flink.api.common.JobExecutionResult;
import org.apache.flink.api.common.accumulators.SerializedListAccumulator;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.java.DataSet;
import org.apache.flink.api.java.ExecutionEnvironment;
import org.apache.flink.api.java.Utils;
import org.apache.flink.util.AbstractID;

import java.util.ArrayList;
import java.util.List;

/**
 * DataSet.collect() always submits the job under the name
 * "Flink Java Job at &lt;timestamp&gt;" with no way to override it.
 * This re-implements collect() using the same accumulator mechanism,
 * but routes through env.execute(jobName) so the Flink UI shows a
 * meaningful name instead.
 */
public final class FlinkJobUtils {

    private FlinkJobUtils() {}

    public static <T> List<T> collect(DataSet<T> dataset, ExecutionEnvironment env, String jobName)
            throws Exception {
        TypeSerializer<T> serializer = dataset.getType().createSerializer(env.getConfig());
        String id = new AbstractID().toString();
        dataset.output(new Utils.CollectHelper<>(id, serializer)).name("collect()");
        JobExecutionResult res = env.execute(jobName);
        ArrayList<byte[]> accResult = res.getAccumulatorResult(id);
        if (accResult == null) {
            throw new RuntimeException("Accumulator result not found for job: " + jobName);
        }
        return SerializedListAccumulator.deserializeList(accResult, serializer);
    }

    public static <T> long count(DataSet<T> dataset, ExecutionEnvironment env, String jobName)
            throws Exception {
        String id = new AbstractID().toString();
        dataset.output(new Utils.CountHelper<T>(id)).name("count()");
        JobExecutionResult res = env.execute(jobName);
        return (Long) res.getAccumulatorResult(id);
    }
}
