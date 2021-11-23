#coding=UTF-8
import random
import bisect
import collections

def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result

def choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]


root_list = ['int','op']
op_list = ["add","minus","multiply","divide"]


p_list = []
op_priority = {
    '+':1,
    '-':1,
    '*':2,
    '/':2
}

def need_col(op1,op2):
    if op_priority[op1] < op_priority[op2]:
        return True
    elif op1=='-' and op2=='-':
        return True


class Expression:
    def __init__(self,prob):
        '''
        :param prob: the probability of root being an integer
        '''
        self.root_type = None
        self.lnode = None
        self.rnode = None
        self.op = None
        rate = 0.1
        p_list = [prob,1-prob]
        if choice(root_list,p_list) == 'int':
            self.root = random.randint(1,99)
            self.expr_value = self.root
            self.root_type = "int"
        else:
            self.rnode = Expression(prob+rate)
            self.lnode = Expression(prob+rate)
            self.op = random.choice(op_list)
            self.root_type = 'op'
            if self.op == 'add':
                self.root = '+'
                self.expr_value = self.lnode.expr_value + self.rnode.expr_value
            elif self.op == 'minus':
                self.root = '-'
                self.expr_value = self.lnode.expr_value - self.rnode.expr_value
            elif self.op == 'multiply':
                self.root = '*'
                self.expr_value = self.lnode.expr_value * self.rnode.expr_value
            elif self.op == 'divide':
                self.root = '/'
                while(self.rnode.expr_value==0) :
                    self.rnode = Expression(prob+rate)

                self.expr_value = self.lnode.expr_value / self.rnode.expr_value



    def __str__(self):
        '''
        考验技术的时候到了 我他吗是不是要自己实现一个stack啊
        :return: string version of the expression
        '''
        if self.root_type == 'int':
            return str(self.root)
        l_expr = ''
        r_expr = ''
        if self.lnode.root_type=='int' or op_priority[self.lnode.root] >= op_priority[self.root]:
            l_expr = str(self.lnode)
        else :
            l_expr = self.lnode.str_with_columns()

        if self.rnode.root_type=='int' or op_priority[self.rnode.root] > op_priority[self.root]:
            r_expr = str(self.rnode)
        else :
            r_expr = self.rnode.str_with_columns()

        return l_expr + self.root + r_expr

    def str_with_columns(self):
        return '(' + str(self) + ')'

    def print_tree(self):
        print(self.root,self.root_type)
        if self.lnode is not None :
            self.lnode.print_tree()

        if self.rnode is not None :
            self.rnode.print_tree()
def gen_expression(target_path,expr_num=100000,max_len = 100):
    '''
    随机生成一个四则运算整数表达式和计算结果，写到对应路径的txt中
    直接用表达式树生成？

    :param max_len:
    :return:
    '''

    with open(target_path,'w') as f:

        for i in range(expr_num):
            expr = Expression(0.2)


if __name__ == '__main__':
    with open("data.txt",'w') as f:
        for i in range(0,10000):
            if i % 1000 ==0 : print("!")
            expr = Expression(0.2)
            x = 0
            try:
                x = eval(str(expr))
            except ZeroDivisionError:
                continue
            if len(str(expr)) > 25 : continue
            if expr.root_type=='int':continue
            f.write(str(expr)+'\n')
            f.write(str(expr.expr_value)+'\n')

# wrong case :(4529)*((1912)*(7750))-1760-3957+8241

