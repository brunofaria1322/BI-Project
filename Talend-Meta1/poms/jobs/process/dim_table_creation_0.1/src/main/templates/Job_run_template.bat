%~d0
cd %~dp0
java -Dtalend.component.manager.m2.repository="%cd%/../lib" -Xms4096M -Xmx4096M -Dfile.encoding=UTF-8 -cp .;../lib/routines.jar;../lib/log4j-slf4j-impl-2.12.1.jar;../lib/log4j-api-2.12.1.jar;../lib/log4j-core-2.12.1.jar;../lib/OneWireAPI.jar;../lib/postgresql-42.2.9.jar;../lib/talendcsv.jar;../lib/crypto-utils.jar;../lib/talend_file_enhanced_20070724.jar;../lib/slf4j-api-1.7.25.jar;../lib/dom4j-2.1.1.jar;dim_table_creation_0_1.jar; bi.dim_table_creation_0_1.Dim_Table_Creation  %*