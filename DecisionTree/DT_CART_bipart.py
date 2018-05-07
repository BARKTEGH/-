# -*- codging:utf-8 -*-
'''
2018-05-07
come from github :https://github.com/michaeldorner/DecisionTrees/blob/master/03_Python%20Code/implementation.py
采用二分法对数据进行划分
最优属性的计算采用 Gini系数
剪枝采用香农熵来作为模型对训练数据的预测误差
'''

import csv
import collections

#树的节点
class DecisionTree:
	"""Binary tree implementation with true and false branch. """
	def __init__(self, col=-1, value=None, trueBranch=None, falseBranch=None, results=None):
		self.col = col
		self.value = value
		self.trueBranch = trueBranch
		self.falseBranch = falseBranch
		self.results = results # None for nodes, not None for leaves

#对数据进行二分
def divideSet(rows, column, value):
	splittingFunction = None
	if isinstance(value, int) or isinstance(value, float): # for int and float values
		splittingFunction = lambda row : row[column] >= value
	else: # for strings
		splittingFunction = lambda row : row[column] == value
	list1 = [row for row in rows if splittingFunction(row)]
	list2 = [row for row in rows if not splittingFunction(row)]
	return (list1, list2)

#计算数据的标签（类别）
#返回字典{类别1：个数，...}
def uniqueCounts(rows):
	results = {}
	for row in rows:
		r = row[-1]
		if r not in results: results[r] = 0
		results[r] += 1
	return results

#计算数据集的香农熵
def entropy(rows):
	from math import log
	log2 = lambda x: log(x)/log(2)
	results = uniqueCounts(rows)
	entr = 0.0
	for r in results:
		p = float(results[r])/len(rows)
		entr -= p*log2(p)
	return entr

#计算数据集的Gini系数
#注意：这里没有1-x，因为想在生成决策树与信息增益统一使用，并使用最大化信息增益。
def gini(rows):
	total = len(rows)
	counts = uniqueCounts(rows)
	imp = 0.0

	for k1 in counts:
		p1 = float(counts[k1])/total
		for k2 in counts:
			if k1 == k2: continue
			p2 = float(counts[k2])/total
			imp += p1*p2
	return imp

#
def variance(rows):
	if len(rows) == 0: return 0
	data = [float(row[len(row) - 1]) for row in rows]
	mean = sum(data) / len(data)

	variance = sum([(d-mean)**2 for d in data]) / len(data)
	return variance

#生成决策树
def growDecisionTreeFrom(rows, evaluationFunction=entropy):
	"""Grows and then returns a binary decision tree.
	evaluationFunction: entropy or gini"""

	if len(rows) == 0: return DecisionTree()
	currentScore = evaluationFunction(rows)
	bestGain = 0.0
	bestAttribute = None
	bestSets = None
	columnCount = len(rows[0]) - 1  # last column is the result/target column，特征属性数量
	#对所有属性用 evaluationFunction方法计算出最优属性和最优划分
	for col in range(0, columnCount):
		columnValues = set([row[col] for row in rows])#保证不重复元素，避免重复计算相同取值的属性值
		for value in columnValues:
			(set1, set2) = divideSet(rows, col, value)
			# Gain -- Entropy or Gini
			p = float(len(set1)) / len(rows)
			gain = currentScore - p*evaluationFunction(set1) - (1-p)*evaluationFunction(set2)
			if gain>bestGain and len(set1)>0 and len(set2)>0:
				bestGain = gain
				bestAttribute = (col, value)
				bestSets = (set1, set2)
    #根据最优属性进行划分，迭代生成决策树
	if bestGain > 0:
		trueBranch = growDecisionTreeFrom(bestSets[0])
		falseBranch = growDecisionTreeFrom(bestSets[1])
		return DecisionTree(col=bestAttribute[0], value=bestAttribute[1], trueBranch=trueBranch, falseBranch=falseBranch)
	else:
		return DecisionTree(results=uniqueCounts(rows))

#剪枝，采用evaluationFunction方法
def prune(tree, minGain, evaluationFunction=entropy, notify=False):
	"""Prunes the obtained tree according to the minimal gain (entropy or Gini). """
	# recursive call for each branch，非叶节点
	if tree.trueBranch.results == None: prune(tree.trueBranch, minGain, evaluationFunction, notify)
	if tree.falseBranch.results == None: prune(tree.falseBranch, minGain, evaluationFunction, notify)

	# merge leaves (potentionally)
	if tree.trueBranch.results != None and tree.falseBranch.results != None:#两个子节点都为叶节点
		tb, fb = [], []
		for v, c in tree.trueBranch.results.items(): tb += [[v]] * c
		for v, c in tree.falseBranch.results.items(): fb += [[v]] * c
		p = float(len(tb)) / len(tb + fb)
		delta = evaluationFunction(tb+fb) - p*evaluationFunction(tb) - (1-p)*evaluationFunction(fb)
		#如果合并的数据集的香农熵下减去分支后的信息增益小于阈值，合并分支
		if delta < minGain:
			if notify: print('A branch was pruned: gain = %f' % delta)
			tree.trueBranch, tree.falseBranch = None, None
			tree.results = uniqueCounts(tb + fb)

def classify(observations, tree, dataMissing=False):
	"""Classifies the observationss according to the tree.
	dataMissing: true or false if data are missing or not. """

	def classifyWithoutMissingData(observations, tree):
		if tree.results != None:  # leaf
			return tree.results
		else:
			v = observations[tree.col]
			branch = None
			if isinstance(v, int) or isinstance(v, float):
				if v >= tree.value: branch = tree.trueBranch
				else: branch = tree.falseBranch
			else:
				if v == tree.value: branch = tree.trueBranch
				else: branch = tree.falseBranch
		return classifyWithoutMissingData(observations, branch)


	def classifyWithMissingData(observations, tree):
		if tree.results != None:  # leaf
			return tree.results
		else:
			v = observations[tree.col]
			if v == None:
				tr = classifyWithMissingData(observations, tree.trueBranch)
				fr = classifyWithMissingData(observations, tree.falseBranch)
				tcount = sum(tr.values())
				fcount = sum(fr.values())
				tw = float(tcount)/(tcount + fcount)
				fw = float(fcount)/(tcount + fcount)
				result = collections.defaultdict(int) # Problem description: http://blog.ludovf.net/python-collections-defaultdict/
				for k, v in tr.items(): result[k] += v*tw
				for k, v in fr.items(): result[k] += v*fw
				return dict(result)
			else:
				branch = None
				if isinstance(v, int) or isinstance(v, float):
					if v >= tree.value: branch = tree.trueBranch
					else: branch = tree.falseBranch
				else:
					if v == tree.value: branch = tree.trueBranch
					else: branch = tree.falseBranch
			return classifyWithMissingData(observations, branch)
	# function body
	if dataMissing:
		return classifyWithMissingData(observations, tree)
	else:
		return classifyWithoutMissingData(observations, tree)


def plot(decisionTree):
    """Plots the obtained decision tree. """

    def toString(decisionTree, indent=''):
        if decisionTree.results != None:  # leaf node
            return str(decisionTree.results)
        else:
            if isinstance(decisionTree.value, int) or isinstance(decisionTree.value, float):
                decision = 'Column %s: x >= %s?' % (decisionTree.col, decisionTree.value)
            else:
                decision = 'Column %s: x == %s?' % (decisionTree.col, decisionTree.value)
            trueBranch = indent + 'yes -> ' + toString(decisionTree.trueBranch, indent + '\t\t')
            falseBranch = indent + 'no  -> ' + toString(decisionTree.falseBranch, indent + '\t\t')
            return (decision + '\n' + trueBranch + '\n' + falseBranch)

    print(toString(decisionTree))


def loadCSV(file):
    """Loads a CSV file and converts all floats and ints into basic datatypes."""

    def convertTypes(s):
        s = s.strip()
        try:
            return float(s) if '.' in s else int(s)
        except ValueError:
            return s

    reader = csv.reader(open(file, 'rt'))
    return [[convertTypes(item) for item in row] for row in reader]


if __name__ == '__main__':

    # Select the example you want to classify
    example = 0

    # All examples do the following steps:
    # 	1. Load training data
    # 	2. Let the decision tree grow
    # 	4. Plot the decision tree
    # 	5. classify without missing data
    # 	6. Classifiy with missing data
    # 	(7.) Prune the decision tree according to a minimal gain level
    # 	(8.) Plot the pruned tree

    if example == 1:
        # the smaller examples
        trainingData = loadCSV('tbc.csv')  # sorry for not translating the TBC and pneumonia symptoms
        decisionTree = growDecisionTreeFrom(trainingData)
        # decisionTree = growDecisionTreeFrom(trainingData, evaluationFunction=gini) # with gini
        plot(decisionTree)
        print(classify(['ohne', 'leicht', 'Streifen', 'normal', 'normal'], decisionTree, dataMissing=False))
        print(
            classify([None, 'leicht', None, 'Flocken', 'fiepend'], decisionTree, dataMissing=True))  # no longer unique

    # Don' forget if you compare the resulting tree with the tree in my presentation: here it is a binary tree!

    else:
        # the bigger example
        trainingData = loadCSV('fishiris.csv')  # demo data from matlab
        decisionTree = growDecisionTreeFrom(trainingData)
        plot(decisionTree)
        prune(decisionTree, 0.5, notify=True)  # notify, when a branch is pruned (one time in this example)
        plot(decisionTree)
        print(classify([6.0, 2.2, 5.0, 1.5], decisionTree))  # dataMissing=False is the default setting
        print(classify([None, None, None, 1.5], decisionTree, dataMissing=True))  # no longer unique
