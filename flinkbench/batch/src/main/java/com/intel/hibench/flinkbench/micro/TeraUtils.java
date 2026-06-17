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

import org.apache.flink.api.common.io.FileInputFormat;
import org.apache.flink.api.common.io.OutputFormat;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.core.fs.FileInputSplit;

import java.io.IOException;

/**
 * Shared Flink-native I/O formats for TeraSort binary data (100-byte records).
 * Avoids HadoopInputFormat/TeraInputFormat to prevent Hadoop version mismatches
 * on Flink JobManager initialisation.
 */
public class TeraUtils {

    public static final int RECORD_SIZE = 100;
    public static final int KEY_SIZE    = 10;

    /** Reads fixed 100-byte TeraSort records from HDFS files. */
    public static class TeraRecordInputFormat extends FileInputFormat<byte[]> {

        private transient org.apache.hadoop.fs.FSDataInputStream stream;
        private transient long splitEnd;
        private transient long position;
        private transient boolean end;

        @Override
        public void open(FileInputSplit split) throws IOException {
            org.apache.hadoop.conf.Configuration conf = new org.apache.hadoop.conf.Configuration();
            org.apache.hadoop.fs.Path hp =
                new org.apache.hadoop.fs.Path(split.getPath().toString());
            org.apache.hadoop.fs.FileSystem fs = hp.getFileSystem(conf);

            long alignedStart = (split.getStart() / RECORD_SIZE) * RECORD_SIZE;
            long rawEnd = split.getStart() + split.getLength();
            splitEnd = ((rawEnd + RECORD_SIZE - 1) / RECORD_SIZE) * RECORD_SIZE;

            stream = fs.open(hp);
            stream.seek(alignedStart);
            position = alignedStart;
            end = (position >= splitEnd);
        }

        @Override
        public boolean reachedEnd() {
            return end;
        }

        @Override
        public byte[] nextRecord(byte[] reuse) throws IOException {
            byte[] record = (reuse != null && reuse.length == RECORD_SIZE)
                ? reuse : new byte[RECORD_SIZE];
            int bytesRead = 0;
            while (bytesRead < RECORD_SIZE) {
                int n = stream.read(record, bytesRead, RECORD_SIZE - bytesRead);
                if (n < 0) { end = true; return null; }
                bytesRead += n;
            }
            position += RECORD_SIZE;
            if (position >= splitEnd) end = true;
            return record;
        }

        @Override
        public void close() throws IOException {
            if (stream != null) { stream.close(); stream = null; }
        }
    }

    /**
     * Writes raw 100-byte records to HDFS.
     * Directly uses Hadoop FS — avoids FileOutputFormat's double-stream conflict.
     */
    public static class TeraRecordOutputFormat implements OutputFormat<byte[]> {

        private final String outputDir;
        private transient org.apache.hadoop.fs.FSDataOutputStream out;

        public TeraRecordOutputFormat(String outputDir) {
            this.outputDir = outputDir;
        }

        @Override
        public void configure(Configuration parameters) {}

        @Override
        public void open(int taskNumber, int numTasks) throws IOException {
            org.apache.hadoop.conf.Configuration conf = new org.apache.hadoop.conf.Configuration();
            org.apache.hadoop.fs.Path dir = new org.apache.hadoop.fs.Path(outputDir);
            org.apache.hadoop.fs.FileSystem fs = dir.getFileSystem(conf);
            fs.mkdirs(dir);
            org.apache.hadoop.fs.Path file = new org.apache.hadoop.fs.Path(
                outputDir + "/part-" + String.format("%05d", taskNumber));
            out = fs.create(file, true);
        }

        @Override
        public void writeRecord(byte[] record) throws IOException {
            out.write(record);
        }

        @Override
        public void close() throws IOException {
            if (out != null) { out.close(); out = null; }
        }
    }

    /** Returns the 10-byte key of a record as a hex string (lexicographically sortable). */
    public static String keyHex(byte[] record) {
        StringBuilder sb = new StringBuilder(KEY_SIZE * 2);
        for (int i = 0; i < KEY_SIZE; i++) {
            sb.append(String.format("%02x", record[i] & 0xFF));
        }
        return sb.toString();
    }
}
