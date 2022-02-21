import json

import base
import os
import sys

def main(): # GENERATE MANY SQLS, cost range between[1w, 10w].
    query_to_path = ""
    para = sys.argv
    # print(para)
    dbname = para[1]
    ctype = para[2]         # cost/card
    mtype = para[3]         # point/range
    N = int(para[4])
    cur_path = os.path.abspath('.')
    db_path = os.path.join(cur_path, dbname ,'logfile')
    if mtype == 'point':
        # print('enter point')
        pc = int(para[5])
        log_path = db_path + '/' + '{}_pc{}_N{}'.format(ctype, pc, N)
        query_to_path = db_path + '/' + '{}_pc{}_N{}_tmp'.format(ctype, pc, N)
    elif mtype == 'range':
        rc = (int(para[5]), int(para[6]))
        # print(db_path)
        log_path = os.path.join(db_path,'{}_rc{}_{}_N{}'.format(ctype, rc[0], rc[1], N))
        query_to_path = os.path.join(db_path,'{}_rc{}_{}_N{}_tmp'.format(ctype, rc[0], rc[1], N))
        #print(log_path)
    else:
        print("error")
    db, cursor = base.connect_server("tpch", "postgresql")
    queries = []
    with open(query_to_path, 'r') as f:
        queries.extend(f.read().split('\n')) # queries = [sql, sql, sql, ...]
        query_id = 0
        for query in queries:
            if query == "":
                continue
            query_id += 1
            explain_analyze_query = "explain (analyze, format json) " + query
            cursor.execute(explain_analyze_query)
            result = cursor.fetchall()[0][0][0]['Plan']
            # print(explain_analyze_query+"\n")
            # print(result)
            with open("testsql/sql_"+str(query_id)+".sql","w") as f1:
                f1.write(explain_analyze_query)
            with open("testsql/analyze_resault_"+str(query_id)+".json","w") as f2:
                json.dump(result,f2)
def gen_sql_tags():
    id = 1
    while (os.path.exists("testsql/sql_"+str(id)+".sql")):
        with open("testsql/analyze_resault_"+str(id)+".json", "r") as f:
            analyzed_plan = json.load(f)
            time = analyzed_plan["Actual Total Time"]
            with open("testsql/sql_time_"+str(id)+".json","w") as f2:
                dic = {"time":float(time)}
                json.dump(dic, f2)
        id = id + 1

if __name__ == '__main__':
    # main()
    gen_sql_tags()