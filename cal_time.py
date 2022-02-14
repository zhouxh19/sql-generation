import os
import time
from copy import copy

import pymysql

import base
import sys
import psycopg2
from enum import IntEnum

# def cal_point_time(dbname, pc, error, N, type, log_path, query_to_path):

def erase_endl(sql):
    sql_n = copy(sql)
    while not sql_n[0].isalpha():
        sql_n = sql_n[1:]
    return sql_n

class DataType(IntEnum):
    VALUE = 0
    TIME = 1
    CHAR = 2

def transfer_field_type(database_type, server):
    data_type = list()
    if server == 'mysql':
        data_type = [['int', 'tinyint', 'smallint', 'mediumint', 'bigint', 'float', 'double', 'decimal'],
                     ['date', 'time', 'year', 'datetime', 'timestamp']]
        database_type = database_type.lower().split('(')[0]
    elif server == 'postgresql':
        data_type = [['integer', 'numeric'],
                     ['date']]
    if database_type in data_type[0]:
        return DataType.VALUE.value
    elif database_type in data_type[1]:
        return DataType.TIME.value
    else:
        return DataType.CHAR.value

def connect_server(dbname, server_name):
    if server_name == 'mysql':
        db = pymysql.connect(host="localhost", user="root", passwd="", db=dbname, charset="utf8")
        cursor = db.cursor()
        return db, cursor
    elif server_name == 'postgresql':
        db = psycopg2.connect(database=dbname, user="lixizhang", password="xi10261026zhang", host="166.111.5.177", port="5433")
        cursor = db.cursor()
        return db, cursor
    else:
        print('数据库连接不上...')
        return


def get_table_structure(cursor, server):
    """
    schema: {table_name: {field_name {'DataType', 'keytype'}}}
    :param cursor:
    :return:
    """
    if server == 'mysql':
        cursor.execute('SHOW TABLES')
        tables = cursor.fetchall()
        schema = {}
        for table_info in tables:
            table_name = table_info[0]
            sql = 'SHOW COLUMNS FROM ' + table_name
            cursor.execute(sql)
            columns = cursor.fetchall()
            schema[table_name] = {}
            for col in columns:
                schema[table_name][col[0]] = [transfer_field_type(col[1], server), col[3]]
            return schema
    elif server == 'postgresql':
        cursor.execute('SELECT table_name FROM information_schema.tables WHERE table_schema = \'public\';')
        tables = cursor.fetchall()
        schema = {}
        for table_info in tables:
            table_name = table_info[0]
            sql = 'SELECT column_name, data_type FROM information_schema.columns WHERE table_name = \'' + table_name + '\';'
            cursor.execute(sql)
            columns = cursor.fetchall()
            schema[table_name] = {}
            for col in columns:
                if col[1] == "integer":
                    schema[table_name][col[0]] = [transfer_field_type(col[1], server)]
        # table_name = tables[0][0]
        # sql = 'SELECT column_name, data_type FROM information_schema.columns WHERE table_name = \'' + table_name + '\';'
        # cursor.execute(sql)
        # columns = cursor.fetchall()
        # schema[table_name] = {}
        # for col in columns:
        #     schema[table_name][col[0]] = [transfer_field_type(col[1], server)]
        return schema


# recover the SQL based on the selected parameter
def generate_sql(template, predicates):
    sql = template
    for x,p in enumerate(predicates):
        sql = sql.replace("[{}]".format(x), predicates[x]["col"] + predicates[x]["op"] + str(predicates[x]["opand"]))
    return sql

def fetch_predicates(sql):
    sql = sql.split()
    predicates = []
    for s in sql:
        if "<" in s:
            s = s.split("<")
            predicates.append({"col":s[0], "op":"<", "opand": float(s[1])})

    return predicates


def compute_grad(best_sql, new_sql, type, target_value):
    if type == "cost":
        state, res = base.get_evaluate_query_info("tpch", best_sql)
        if state == 0:
            return -1,-1

        card1 = res["total_cost"]
        card1 = card1 if card1 != 0 else 1
        if isinstance(target_value, int): # value
            err1 = min(card1 / target_value, target_value / card1)
        else: # range
            lower_bound = target_value[0]
            upper_bound = target_value[1]

            if lower_bound<=card1 and card1<=upper_bound:
                err1 = 1
            else:
                err1_uper = min(card1 / lower_bound, lower_bound / card1)
                err1_lower = min(card1 / upper_bound, upper_bound / card1)

                err1 = max(err1_uper, err1_lower)

        state, res = base.get_evaluate_query_info("tpch", new_sql)
        card2 = res["total_cost"]
        card2 = card2 if card2 != 0 else 1
        if isinstance(target_value, int): # value
            err2 = min(card2 / target_value, target_value / card2)
        else: # range
            lower_bound = target_value[0]
            upper_bound = target_value[1]

            if lower_bound <= card2 and card2 <= upper_bound:
                err2 = 1
            else:
                err2_uper = min(card2 / lower_bound, lower_bound / card2)
                err2_lower = min(card2 / upper_bound, upper_bound / card2)

                err2 = max(err2_uper, err2_lower)

        # print("[+/-step]", card1, card2, target_value)

        # print("[+/-step]", card1, card2, target_value)

        return err2, err1

    elif type == "card":
        state, res = base.get_evaluate_query_info("tpch", best_sql)
        if state == 0:
            return -1, -1

        card1 = res["e_cardinality"]
        card1 = card1 if card1 != 0 else 1
        if isinstance(target_value, int): # value
            err1 = min(card1 / target_value, target_value / card1)
        else: # range
            lower_bound = target_value[0]
            upper_bound = target_value[1]

            if lower_bound<=card1 and card1<=upper_bound:
                err1 = 1
            else:
                err1_uper = min(card1 / lower_bound, lower_bound / card1)
                err1_lower = min(card1 / upper_bound, upper_bound / card1)

                err1 = max(err1_uper, err1_lower)


        state, res = base.get_evaluate_query_info("tpch", new_sql)
        card2 = res["e_cardinality"]
        card2 = card2 if card2 != 0 else 1
        if isinstance(target_value, int): # value
            err2 = min(card2 / target_value, target_value / card2)
        else: # range
            lower_bound = target_value[0]
            upper_bound = target_value[1]

            if lower_bound <= card2 and card2 <= upper_bound:
                err2 = 1
            else:
                err2_uper = min(card2 / lower_bound, lower_bound / card2)
                err2_lower = min(card2 / upper_bound, upper_bound / card2)

                err2 = max(err2_uper, err2_lower)

        # print("[+/-step]", card1, card2, target_value)

        return err2, err1

    return -1, -1

def extract_predicates(sql, schema, ops):
    tokens = sql.split()
    predicates = [] # [col, left-operand, op, right-operand]

    for i,token in enumerate(tokens):
        if token in ops and i-1>=0 and i+1<=len(tokens)-1: # a predicate
            for table in schema:
                for col in schema[table]:
                    if col in tokens[i-1]:
                        predicates.append({"col": table+'.'+col, "left": tokens[i-1], "op": token, "right": tokens[i+1]})

    return predicates


def select_parameter(x, p, tokens, step, col_range, best_sql):

    col = p["col"]
    op = p["op"]
    opand = float(p["right"]) + step * (-col_range[col]["min"] + col_range[col]["max"]) # random
    opand = int(round(opand, 0))
    # print("opand", str(opand))

    if opand > col_range[col]["max"]:
        opand = col_range[col]["max"]
    elif opand < col_range[col]["min"]:
        opand = col_range[col]["min"]
    #if opand !=0:
    #    print("[value]", opand)
    sql_tokens = best_sql.split()
    # locate the x-th </<=
    i = 0
    for pos,t in enumerate(sql_tokens):
        if t in tokens:
            if i == x:
                sql_tokens[pos+1] = str(opand)
                break
            else:
                i = i + 1
    new_sql = ""
    for t in sql_tokens:
        new_sql = new_sql + " " + t

    return new_sql

def is_required_sql(best_sql, type, pc, error):
    state, res = base.get_evaluate_query_info("tpch", best_sql)
    if state == 0:
        return 0

    if type == "cost":
        cost = res["total_cost"]
        # print("[distance]", cost, pc)
        if isinstance(pc, int):
            if cost >= pc*(1-error) and cost <= pc*(1+error):
                return 1
            else:
                return 0
        else:
            lower_bound = pc[0]
            upper_bound = pc[1]

            # print(lower_bound, upper_bound, cost)

            if lower_bound<=cost and cost<=upper_bound:
                return 1
            else:
                return 0

    elif type == "card":
        card = res["e_cardinality"]
        # print("[distance]", card, pc)

        if isinstance(pc, int):
            if card >= pc * (1 - error) and card <= pc * (1 + error):
                return 1
            else:
                return 0
        else:
            lower_bound = pc[0]
            upper_bound = pc[1]

            print(lower_bound, upper_bound, card)


            if lower_bound <= card and card <= upper_bound:
                return 1
            else:
                return 0

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def hillclimb_generate_queries(dbname, type, target_value, error, num_queries, target_path):

    start_time = time.time()
    # srandomly generated sql templates
    # requirements of the sql templates
        # have predicates (< or <=)
        # the columns in predicates are integer type
    # sql_template = "select partsupp.ps_suppkey from partsupp join part on ps_partkey=p_partkey where [0] and [1] order by partsupp.ps_suppkey ASC;"
    query_templates = []
    with open("./sql-template/sql_templates.log", 'r') as f:
        query_templates.extend(f.read().split(';'))

    db, cursor = connect_server("tpch", "postgresql")
    schema = get_table_structure(cursor, "postgresql")

    ok_queries = set(())
    template_id = 0
    while len(ok_queries) < num_queries and template_id < len(query_templates):
        # print("[id]",template_id)
        template = query_templates[template_id]
        template_id = template_id + 1
        predicates = extract_predicates(template, schema, ["<", "<="])
        # print(predicates) # {"col": table+'.'+col, "left": tokens[i-1], "op": token, "right": tokens[i+1]}

        # generate the value ranges (column, min, max)
        # col_range = {"partsupp.ps_partkey": {"min": 1, "max": 9999}, "partsupp.ps_suppkey": {"min": 1, "max": 9999}}
        col_range = {}
        for predicate in predicates:
            if predicate["col"] not in col_range:
                col = predicate["col"].split(".")
                # print(col)
                cursor.execute("select min("+col[1]+") from "+col[0])
                min_value = cursor.fetchall()[0][0]

                cursor.execute("select max("+col[1]+") from "+col[0])
                max_value = cursor.fetchall()[0][0]
                col_range[predicate['col']] = {"min": min_value, "max": max_value}

        # hill-climbing
        step = 0.8
        tokens = ["<", "<="]
        best_sql = template

        ok = is_required_sql(best_sql, type, target_value, error)
        # print("[current best]", best_sql)
        # orders.o_orderkey <= 5858 and orders.o_shippriority < 0
        while step > 0.0001 and not ok:
        # while step > 0.0001:
            best_predicates = extract_predicates(best_sql, schema, tokens)
            max_grad = 0

            for x, p in enumerate(best_predicates):  # 枚举1层谓词加入到sql语句中
                if not is_number(p["right"]):
                    continue

                # (step)
                # print("[value]", step * (-col_range[p['col']]["min"] + col_range[p['col']]["max"]))
                new_sql = select_parameter(x, p, tokens, step, col_range, best_sql) # 把x和p加入到best_sql中得到new sql.
                err_new, err_best = compute_grad(best_sql, new_sql, type, target_value)
                # print(err_new, err_best)
                grad = (err_new - err_best)
                if grad > max_grad:  # 选择最大梯度下降
                    best_sql = new_sql
                    max_grad = grad
                    ok = is_required_sql(best_sql, type, target_value, error)
                    if err_new == 1:
                        ok = 1
                    # print("[current best]", new_sql)

                # (-step)
                new_sql = select_parameter(x, p, tokens, -step, col_range, best_sql)
                err_new, err_best = compute_grad(best_sql, new_sql, type, target_value)
                grad = (err_new - err_best)
                if grad > max_grad:
                    best_sql = new_sql
                    max_grad = grad
                    ok = is_required_sql(best_sql, type, target_value, error)
                    if err_new == 1:
                        ok = 1
                    # print("[current best]", new_sql)

            if max_grad == 0:
                step = round(step / 2, 5)
                # jump out

        if ok:
            if best_sql not in ok_queries:

                ok_queries.add(best_sql)

                with open(target_path, "a") as f:
                    f.write(erase_endl(best_sql)+";\n")
                #print("? best_sql",best_sql,"!")
                print("time:{};s_count:{};t_count:{}".format(str(time.time() - start_time), len(ok_queries), template_id))
                # print(best_sql)

            if len(ok_queries) % (num_queries/100) == 0:
                print("time:{};s_count:{};t_count:{}\n".format(str(time.time()-start_time), len(ok_queries), template_id))
                with open(target_path, "a") as f:
                    f.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()-start_time), len(ok_queries), template_id))
        elif template_id % (100) == 0:
            print("time:{};s_count:{};t_count:{}\n".format(str(time.time() - start_time), len(ok_queries), template_id))

    print("end")
    exit()


def cal_point_time(dbname, pc, error, N, type, log_path, query_to_path): # pc : point的值(cost = pc-K ~ pc+K)

    print("Generating {} queries meet point {} constraint:{} with acceptable error {} to '{}'".format(N, type, pc,
                                                                                                      error, log_path))
    time.sleep(1)
    satisfied_count = 0
    total_count = 0

    # print("log_path:", log_path)

    if os.path.exists(log_path):
        log = open(log_path, 'r+')
        lines = log.readlines()
        last_line = lines[-1]
        print(last_line)
        satisfied_count = int(last_line.split(';')[1].split(':')[1])
        total_count = int(last_line.split(';')[2].split(':')[1])
    else:
        log = open(log_path, 'w')

    log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
    low_bound = pc*(1 - error)
    up_bound = pc*(1 + error)
    # while satisfied_count < N:satisfied_count

    hillclimb_generate_queries(dbname, type, pc, error, N, query_to_path)  # 用爬山算法生成N个query.
    total_count += N
    queries = []
    with open(query_to_path, 'r') as f:
        queries.extend(f.read().split(';'))
        for query in queries:
            try:
                result, e_info = base.get_evaluate_query_info(dbname, query)
                if result:  # 如果可以估计query的代价
                    if type == "cost":  # 分 cost / card 两种维度
                        if low_bound <= e_info['total_cost'] <= up_bound:
                            satisfied_count += 1
                            if satisfied_count % 100 == 0:
                                log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
                            if total_count % 1000 == 0:
                                print("{}/{} time: {}\n".format(satisfied_count, total_count, str(time.time())))
                    elif type == "card":
                        if low_bound <= e_info['e_cardinality'] <= up_bound:
                            satisfied_count += 1
                            if satisfied_count % 100 == 0:
                                log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
                            if total_count % 1000 == 0:
                                print("{}/{} time: {}\n".format(satisfied_count, total_count, str(time.time())))
                    else:
                        print("error")
                        return
            except Exception as result:
                log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
                log.close()
                print(result)

    log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
    log.close()

def cal_range_time(dbname, rc, error, N, type, log_path, query_to_path):
    print("Generating {} queries meet range {} constraint:[{}, {}] to '{}'".format(N, type, rc[0], rc[1], log_path))
    time.sleep(2)
    satisfied_count = 0
    total_count = 0

    if os.path.exists(log_path):
        log = open(log_path, 'r+')
        lines = log.readlines()
        last_line = lines[-1]
        print(last_line)
        satisfied_count = int(last_line.split(';')[1].split(':')[1])
        total_count = int(last_line.split(';')[2].split(':')[1])
    else:
        log = open(log_path, 'w')

    log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
    low_bound = rc[0]
    up_bound = rc[1]

#    while satisfied_count < N:
    # hillclimb_generate_queries(dbname, pc, error, N, query_to_path)
    hillclimb_generate_queries(dbname, type, (low_bound, up_bound), error, N, query_to_path)

    total_count += N
    queries = []
    with open(query_to_path, 'r') as f:
        queries.extend(f.read().split(';'))
        for query in queries:
            try:
                result, e_info = base.get_evaluate_query_info(dbname, query)
                if result:
                    if type == "cost":
                        if low_bound <= e_info['total_cost'] and e_info['total_cost'] <= up_bound:
                            satisfied_count += 1
                            if satisfied_count % 100 == 0:

                                log.write(
                                    "time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count,
                                                                             total_count))
                            if total_count % 1000 == 0:
                                print("{}/{} time: {}\n".format(satisfied_count, total_count, str(time.time())))
                    elif type == "card":
                        if low_bound <= e_info['e_cardinality'] and e_info['e_cardinality'] <= up_bound:
                            satisfied_count += 1
                            if satisfied_count % 100 == 0:
                                log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count,
                                                                                   total_count))
                            if total_count % 1000 == 0:
                                print("{}/{} time: {}\n".format(satisfied_count, total_count, str(time.time())))
                    else:
                        print("error")
                        return
            except Exception as result:
                log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
                log.close()
                print(result)
    log.write("time:{};s_count:{};t_count:{}\n".format(str(time.time()), satisfied_count, total_count))
    log.close()

# cal_point_time('tpch', pc=0, error=10, N=1000, type='cost')
# cal_point_time('tpch', pc=0, error=10, N=1000, type='card')
# cal_point_time('tpch', pc=1000, error=100, N=1000, type='cost')
# cal_point_time('tpch', pc=1000, error=100, N=1000, type='card')
# cal_point_time('tpch', pc=10000, error=1000, N=1000, type='cost')
# cal_point_time('tpch', pc=10000, error=1000, N=1000, type='card',
#                query_to_path='/home/lixizhang/learnSQL/sqlsmith/tpch/logfile/tmp',
#                log_path='/home/lixizhang/learnSQL/sqlsmith/tpch/logfile/card_pc10000_N1000')

if __name__ == '__main__':
    para = sys.argv
    # print(para)
    dbname = para[1]
    ctype = para[2]         # cost/card
    mtype = para[3]         # point/range
    N = int(para[4])
    cur_path = os.path.abspath('.')
    db_path = os.path.join(cur_path, dbname ,'logfile')
    error = 0.2
    if mtype == 'point':
        # print('enter point')
        pc = int(para[5])
        log_path = db_path + '/' + '{}_pc{}_N{}'.format(ctype, pc, N)
        tmp_path = db_path + '/' + '{}_pc{}_N{}_tmp'.format(ctype, pc, N)
        cal_point_time(dbname=dbname, type=ctype, N=N, pc=pc, error=error, log_path=log_path, query_to_path=tmp_path)
    elif mtype == 'range':
        rc = (int(para[5]), int(para[6]))
        # print(db_path)
        log_path = os.path.join(db_path,'{}_rc{}_{}_N{}'.format(ctype, rc[0], rc[1], N))
        tmp_path = os.path.join(db_path,'{}_rc{}_{}_N{}_tmp'.format(ctype, rc[0], rc[1], N))
        #print(log_path)
        cal_range_time(dbname=dbname, type=ctype, N=N, error=error, rc=rc, log_path=log_path, query_to_path=tmp_path)
    else:
        print("error")
