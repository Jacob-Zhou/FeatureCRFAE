from collections.abc import Iterable

import nltk
from tagger.utils.logging import get_logger, progress_bar

logger = get_logger(__name__)


class Form(object):
    """
    A Transform object corresponds to a specific data format.
    It holds several instances of data fields that provide instructions for preprocessing and numericalizing, etc.

    Attributes:
        training (bool, default: True):
            Set the object in training mode.
            If False, some data fields not required for predictions won't be returned.
    """

    fields = []

    def __init__(self):
        self.training = True

    def __call__(self, sentences):
        pairs = dict()
        for field in self:
            if not isinstance(field, Iterable):
                field = [field]
            for f in field:
                if f is not None and not f.is_placeholder:
                    pairs[f] = f.transform(
                        [getattr(i, f.name) for i in sentences])

        return pairs

    def __getitem__(self, index):
        return getattr(self, self.fields[index])

    def train(self, training=True):
        self.training = training

    def eval(self):
        self.train(False)

    def append(self, field):
        self.fields.append(field.name)
        setattr(self, field.name, field)

    @property
    def src(self):
        raise AttributeError

    @property
    def tgt(self):
        raise AttributeError

    def save(self, path, sentences):
        with open(path, 'w') as f:
            f.write('\n'.join([str(i) for i in sentences]) + '\n')


class Sentence(object):
    """
    A Sentence object holds a sentence with regard to specific data format.
    """
    def __init__(self, transform):
        self.transform = transform

        # mapping from each nested field to their proper position
        self.maps = dict()
        # names of each field
        self.keys = set()
        # values of each position
        self.values = []
        for i, field in enumerate(self.transform):
            if not isinstance(field, Iterable):
                field = [field]
            for f in field:
                if f is not None:
                    self.maps[f.name] = i
                    self.keys.add(f.name)

    def __len__(self):
        return len(self.values[0])

    def __contains__(self, key):
        return key in self.keys

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        else:
            return self.values[self.maps[name]]

    def __setattr__(self, name, value):
        if 'keys' in self.__dict__ and name in self:
            index = self.maps[name]
            if index >= len(self.values):
                self.__dict__[name] = value
            else:
                self.values[index] = value
        else:
            self.__dict__[name] = value

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        self.__dict__.update(state)


class CoNLL(Form):
    """
    The CoNLL object holds ten fields required for CoNLL-X data format.
    Each field is binded with one or more Field objects. For example,
    the FORM field can contain both Field and SubwordField to produce tensors for words and subwords.

    For each sentence, the ten fields are:
        ID:
            Token counter, starting at 1.
        FORM:
            Words in the sentence.
        LEMMA:
            Lemmas or stems (depending on the particular treebank) of words, or underscores if not available.
        CPOS:
            Coarse-grained part-of-speech tags, where the tagset depends on the treebank.
        POS:
            Fine-grained part-of-speech tags, where the tagset depends on the treebank.
        FEATS:
            Unordered set of syntactic and/or morphological features (depending on the particular treebank),
            or underscores if not available.
        HEAD:
            Heads of the tokens, which are either values of ID or zeros.
        DEPREL:
            Dependency relations to the HEAD.
        PHEAD:
            Projective heads of tokens, which are either values of ID or zeros, or underscores if not available.
        PDEPREL:
            Dependency relations to the PHEAD, or underscores if not available.

    References:
    - Sabine Buchholz and Erwin Marsi (CoNLL'06)
      CoNLL-X Shared Task on Multilingual Dependency Parsing
      https://www.aclweb.org/anthology/W06-2920/
    """

    fields = [
        'ID', 'FORM', 'LEMMA', 'CPOS', 'POS', 'FEATS', 'HEAD', 'DEPREL',
        'PHEAD', 'PDEPREL'
    ]

    def __init__(self,
                 ID=None,
                 FORM=None,
                 LEMMA=None,
                 CPOS=None,
                 POS=None,
                 FEATS=None,
                 HEAD=None,
                 DEPREL=None,
                 PHEAD=None,
                 PDEPREL=None):
        super().__init__()

        self.ID = ID
        self.FORM = FORM
        self.LEMMA = LEMMA
        self.CPOS = CPOS
        self.POS = POS
        self.FEATS = FEATS
        self.HEAD = HEAD
        self.DEPREL = DEPREL
        self.PHEAD = PHEAD
        self.PDEPREL = PDEPREL

    @property
    def src(self):
        return self.FORM, self.CPOS

    @property
    def tgt(self):
        return self.HEAD, self.DEPREL

    @classmethod
    def get_arcs(cls, sequence):
        return [int(i) for i in sequence]

    @classmethod
    def get_sibs(cls, sequence):
        sibs = [-1] * (len(sequence) + 1)
        heads = [0] + [int(i) for i in sequence]

        for i in range(1, len(heads)):
            hi = heads[i]
            for j in range(i + 1, len(heads)):
                hj = heads[j]
                di, dj = hi - i, hj - j
                if hi >= 0 and hj >= 0 and hi == hj and di * dj > 0:
                    if abs(di) > abs(dj):
                        sibs[i] = j
                    else:
                        sibs[j] = i
                    break
        return sibs[1:]

    @classmethod
    def toconll(cls, tokens):
        """
        Convert a list of tokens to a string in CoNLL-X format.
        Missing fields are filled with underscores.

        Args:
            tokens (List[str] or List[tuple]):
                This can be either a list of words or word/pos pairs.

        Returns:
            a string in CoNLL-X format.

        Examples::
            >>> print(CoNLL.toconll(['I', 'saw', 'Sarah', 'with', 'a', 'telescope']))
            1       I       _       _       _       _       _       _       _       _
            2       saw     _       _       _       _       _       _       _       _
            3       Sarah   _       _       _       _       _       _       _       _
            4       with    _       _       _       _       _       _       _       _
            5       a       _       _       _       _       _       _       _       _
            6       telescope       _       _       _       _       _       _       _       _
        """

        if isinstance(tokens[0], str):
            s = '\n'.join([
                f"{i}\t{word}\t" + '\t'.join(['_'] * 8)
                for i, word in enumerate(tokens, 1)
            ])
        else:
            s = '\n'.join([
                f"{i}\t{word}\t_\t{tag}\t" + '\t'.join(['_'] * 6)
                for i, (word, tag) in enumerate(tokens, 1)
            ])
        return s + '\n'

    @classmethod
    def isprojective(cls, sequence):
        """
        Check if the dependency tree is projective.
        This also works for partial annotation.

        Besides the obvious crossing arcs, the examples below illustrate two non-projective cases
        that are hard to detect in the scenario of partial annotation.

        Args:
            sequence (List[int]):
                A list of head indices.

        Returns:
            True if the tree is projective, False otherwise.

        Examples::
            >>> CoNLL.isprojective([2, -1, 1])  # -1 denotes un-annotated cases
            False
            >>> CoNLL.isprojective([3, -1, 2])
            False
        """

        pairs = [(h, d) for d, h in enumerate(sequence, 1) if h >= 0]
        for i, (hi, di) in enumerate(pairs):
            for hj, dj in pairs[i + 1:]:
                (li, ri), (lj, rj) = sorted([hi, di]), sorted([hj, dj])
                if li <= hj <= ri and hi == dj:
                    return False
                if lj <= hi <= rj and hj == di:
                    return False
                if (li < lj < ri
                        or li < rj < ri) and (li - lj) * (ri - rj) > 0:
                    return False
        return True

    @classmethod
    def istree(cls, sequence, proj=False, multiroot=False):
        """
        Check if the arcs form an valid dependency tree.

        Args:
            sequence (List[int]):
                A list of head indices.
            proj (bool, default: False):
                If True, requires the tree to be projective.
            multiroot (bool, default: True):
                If False, requires the tree to contain only a single root.

        Returns:
            True if the arcs form an valid tree, False otherwise.

        Examples::
            >>> CoNLL.istree([3, 0, 0, 3], multiroot=True)
            True
            >>> CoNLL.istree([3, 0, 0, 3], proj=True)
            False
        """

        from tagger.utils.alg import tarjan
        if proj and not cls.isprojective(sequence):
            return False
        n_roots = sum(head == 0 for head in sequence)
        if n_roots == 0:
            return False
        if not multiroot and n_roots > 1:
            return False
        if any(i == head for i, head in enumerate(sequence, 1)):
            return False
        return next(tarjan(sequence), None) is None

    def load(self, data, proj=False, max_len=None, **kwargs):
        """
        Load data in CoNLL-X format.
        Also support for loading data from CoNLL-U file with comments and non-integer IDs.

        Args:
            data (List[List] or str):
                A list of instances or a filename.
            proj (bool, default: False):
                If True, discard all non-projective sentences.
            max_len (int, default: None):
                Sentences exceeding the length will be discarded.

        Returns:
            A list of CoNLLSentence instances.
        """

        if isinstance(data, str):
            with open(data, 'r') as f:
                lines = [line.strip() for line in f]
        elif isinstance(data, dict):
            # {file_name: weight}
            lines = []
            for k, v in data.items():
                with open(k, 'r') as f:
                    if not isinstance(v, int):
                        v = 1
                    lines += ([line.strip() for line in f] + ['']) * v
        else:
            data = [data] if isinstance(data[0], str) else data
            lines = '\n'.join([self.toconll(i) for i in data]).split('\n')

        i, start, sentences = 0, 0, []
        for line in progress_bar(lines):
            if not line:
                sentences.append(CoNLLSentence(self, lines[start:i]))
                start = i + 1
            i += 1
        if proj:
            sentences = [
                i for i in sentences
                if self.isprojective(list(map(int, i.arcs)))
            ]
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]

        return sentences


class CoNLLSentence(Sentence):
    """
    Sencence in CoNLL-X format.

    Args:
        transform (CoNLL):
            A CoNLL object.
        lines (List[str]):
            A list of strings composing a sentence in CoNLL-X format.
            Comments and non-integer IDs are permitted.

    Examples::
        >>> lines = ['# text = But I found the location wonderful and the neighbors very kind.',
                     '1\tBut\t_\t_\t_\t_\t_\t_\t_\t_',
                     '2\tI\t_\t_\t_\t_\t_\t_\t_\t_',
                     '3\tfound\t_\t_\t_\t_\t_\t_\t_\t_',
                     '4\tthe\t_\t_\t_\t_\t_\t_\t_\t_',
                     '5\tlocation\t_\t_\t_\t_\t_\t_\t_\t_',
                     '6\twonderful\t_\t_\t_\t_\t_\t_\t_\t_',
                     '7\tand\t_\t_\t_\t_\t_\t_\t_\t_',
                     '7.1\tfound\t_\t_\t_\t_\t_\t_\t_\t_',
                     '8\tthe\t_\t_\t_\t_\t_\t_\t_\t_',
                     '9\tneighbors\t_\t_\t_\t_\t_\t_\t_\t_',
                     '10\tvery\t_\t_\t_\t_\t_\t_\t_\t_',
                     '11\tkind\t_\t_\t_\t_\t_\t_\t_\t_',
                     '12\t.\t_\t_\t_\t_\t_\t_\t_\t_']
        >>> sentence = CoNLLSentence(transform, lines)  # fields in transform are built from ptb.
        >>> sentence.arcs = [3, 3, 0, 5, 6, 3, 6, 9, 11, 11, 6, 3]
        >>> sentence.rels = ['cc', 'nsubj', 'root', 'det', 'nsubj', 'xcomp',
                             'cc', 'det', 'dep', 'advmod', 'conj', 'punct']
        >>> sentence
        # text = But I found the location wonderful and the neighbors very kind.
        1       But     _       _       _       _       3       cc      _       _
        2       I       _       _       _       _       3       nsubj   _       _
        3       found   _       _       _       _       0       root    _       _
        4       the     _       _       _       _       5       det     _       _
        5       location        _       _       _       _       6       nsubj   _       _
        6       wonderful       _       _       _       _       3       xcomp   _       _
        7       and     _       _       _       _       6       cc      _       _
        7.1     found   _       _       _       _       _       _       _       _
        8       the     _       _       _       _       9       det     _       _
        9       neighbors       _       _       _       _       11      dep     _       _
        10      very    _       _       _       _       11      advmod  _       _
        11      kind    _       _       _       _       6       conj    _       _
        12      .       _       _       _       _       3       punct   _       _
    """
    def __init__(self, transform, lines):
        super().__init__(transform)

        self.values = []
        # record annotations for post-recovery
        self.annotations = dict()

        for i, line in enumerate(lines):
            value = line.split('\t')
            if value[0].startswith('#') or not value[0].isdigit():
                self.annotations[-i - 1] = line
            else:
                self.annotations[len(self.values)] = line
                self.values.append(value)
        self.values = list(zip(*self.values))

    def __repr__(self):
        # cover the raw lines
        merged = {
            **self.annotations,
            **{
                i: '\t'.join(map(str, line))
                for i, line in enumerate(zip(*self.values))
            }
        }
        return '\n'.join(merged.values()) + '\n'


class Tree(Form):
    """
    The Tree object factorize a constituency tree into four fields, each associated with one or more Field objects:
        WORD:
            Words in the sentence.
        POS:
            Part-of-speech tags, or underscores if not available.
        TREE:
            The raw constituency tree in nltk.Tree format.
        CHART:
            The factorized sequence of binarized tree traversed in pre-order.
    """

    root = ''
    fields = ['WORD', 'POS', 'TREE', 'CHART']

    def __init__(self, WORD=None, POS=None, TREE=None, CHART=None):
        super().__init__()

        self.WORD = WORD
        self.POS = POS
        self.TREE = TREE
        self.CHART = CHART

    @property
    def src(self):
        return self.WORD, self.POS, self.TREE

    @property
    def tgt(self):
        return self.CHART,

    @classmethod
    def totree(cls, tokens, root=''):
        """
        Convert a list of tokens to a nltk.Tree.
        Missing fields are filled with underscores.

        Args:
            tokens (List[str] or List[tuple]):
                This can be either a list of words or word/pos pairs.
            root (str, default: ''):
                The root label of the tree.

        Returns:
            a nltk.Tree object.

        Examples::
            >>> print(Tree.totree(['I', 'really', 'love', 'this', 'game'], 'TOP'))
            (TOP (_ I) (_ really) (_ love) (_ this) (_ game))
        """

        if isinstance(tokens[0], str):
            tokens = [(token, '_') for token in tokens]
        tree = ' '.join([f"({pos} {word})" for word, pos in tokens])
        return nltk.Tree.fromstring(f"({root} {tree})")

    @classmethod
    def binarize(cls, tree):
        """
        Conduct binarization over the tree.

        First, the tree is transformed to satisfy Chomsky Normal Form (CNF).
        Here we call the member function `chomsky_normal_form` in nltk.Tree to conduct left-binarization.
        Second, all unary productions in the tree are collapsed.

        Args:
            tree (nltk.Tree):
                the tree to be binarized.

        Returns:
            the binarized tree.

        Examples::
            >>> tree = nltk.Tree.fromstring('''
                                            (TOP
                                              (S
                                                (NP (_ I))
                                                  (ADVP (_ really))
                                                    (VP (_ love) (NP (_ this) (_ game)))))
                                            ''')
            >>> print(Tree.binarize(tree))
            (TOP
              (S
                (S|<> (NP (_ I)) (ADVP (_ really)))
                (VP (VP|<> (_ love)) (NP (NP|<> (_ this)) (NP|<> (_ game))))))
        """

        tree = tree.copy(True)
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, nltk.Tree):
                nodes.extend([child for child in node])
                if len(node) > 1:
                    for i, child in enumerate(node):
                        if not isinstance(child[0], nltk.Tree):
                            node[i] = nltk.Tree(f"{node.label()}|<>", [child])
        tree.chomsky_normal_form('left', 0, 0)
        tree.collapse_unary()

        return tree

    @classmethod
    def factorize(cls, tree, delete_labels=None, equal_labels=None):
        """
        Factorize the tree into a sequence.
        The tree is traversed in pre-order.

        Args:
            tree (nltk.Tree):
                the tree to be factorized.
            delete_labels (Set[str], default: None):
                A set of labels to be ignored. This is used for evaluation.
                If it is a pre-terminal label, delete the word along with the brackets.
                If it is a non-terminal label, just delete the brackets (don't delete childrens).
                In EVALB (https://nlp.cs.nyu.edu/evalb/), the default set is:
                {'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''}
            equal_labels (Dict[str, str], default: None):
                The key-val pairs in the dict are considered equivalent (non-directional). This is used for evaluation.
                The default dict defined in EVALB is: {'ADVP': 'PRT'}

        Returns:
            The sequence of factorized tree.

        Examples::
            >>> tree = nltk.Tree.fromstring('''
                                            (TOP
                                              (S
                                                (NP (_ I))
                                                  (ADVP (_ really))
                                                    (VP (_ love) (NP (_ this) (_ game)))))
                                            ''')
            >>> Tree.factorize(tree)
            [(0, 5, 'TOP'), (0, 5, 'S'), (0, 1, 'NP'), (1, 2, 'ADVP'), (2, 5, 'VP'), (3, 5, 'NP')]
            >>> Tree.factorize(tree, delete_labels={'TOP', 'S1', '-NONE-', ',', ':', '``', "''", '.', '?', '!', ''})
            [(0, 5, 'S'), (0, 1, 'NP'), (1, 2, 'ADVP'), (2, 5, 'VP'), (3, 5, 'NP')]
        """
        def track(tree, i):
            label = tree.label()
            if delete_labels is not None and label in delete_labels:
                label = None
            if equal_labels is not None:
                label = equal_labels.get(label, label)
            if len(tree) == 1 and not isinstance(tree[0], nltk.Tree):
                return (i + 1 if label is not None else i), []
            j, spans = i, []
            for child in tree:
                j, s = track(child, j)
                spans += s
            if label is not None and j > i:
                spans = [(i, j, label)] + spans
            return j, spans

        return track(tree, 0)[1]

    @classmethod
    def build(cls, tree, sequence):
        """
        Build a constituency tree from the sequence. The sequence is generated in pre-order.
        During building the tree, the sequence is de-binarized to the original format (i.e.,
        the suffixes '|<>' are ignored, the collapsed labels are recovered).

        Args:
            tree (nltk.Tree):
                An empty tree providing a base for building a result tree.
            sequence (List[tuple]):
                A list of tuples used for generating a tree.
                Each tuple consits of the indices of left/right span boundaries and label of the span.

        Returns:
            A result constituency tree.

        Examples::
            >>> tree = Tree.totree(['I', 'really', 'love', 'this', 'game'], 'TOP')
            >>> sequence = [(0, 5, 'S'), (0, 2, 'S|<>'), (0, 1, 'NP'), (1, 2, 'ADVP'), (2, 5, 'VP'),
                            (2, 3, 'VP|<>'), (3, 5, 'NP'), (3, 4, 'NP|<>'), (4, 5, 'NP|<>')]
            >>> print(Tree.build(tree, sequence))
            (TOP
              (S
                (NP (_ I))
                  (ADVP (_ really))
                    (VP (_ love) (NP (_ this) (_ game)))))
        """

        root = tree.label()
        leaves = [
            subtree for subtree in tree.subtrees()
            if not isinstance(subtree[0], nltk.Tree)
        ]

        def track(node):
            i, j, label = next(node)
            if j == i + 1:
                children = [leaves[i]]
            else:
                children = track(node) + track(node)
            if label.endswith('|<>'):
                return children
            labels = label.split('+')
            tree = nltk.Tree(labels[-1], children)
            for label in reversed(labels[:-1]):
                tree = nltk.Tree(label, [tree])
            return [tree]

        return nltk.Tree(root, track(iter(sequence)))

    def load(self, data, max_len=None, **kwargs):
        """
        Args:
            data (List[List] or str):
                A list of instances or a filename.
            max_len (int, default: None):
                Sentences exceeding the length will be discarded.

        Returns:
            A list of TreeSentence instances.
        """
        if isinstance(data, str):
            with open(data, 'r') as f:
                trees = [nltk.Tree.fromstring(string) for string in f]
            self.root = trees[0].label()
        else:
            data = [data] if isinstance(data[0], str) else data
            trees = [self.totree(i, self.root) for i in data]

        i, sentences = 0, []
        for tree in progress_bar(trees, leave=False):
            if len(tree) == 1 and not isinstance(tree[0][0], nltk.Tree):
                continue
            sentences.append(TreeSentence(self, tree))
            i += 1
        if max_len is not None:
            sentences = [i for i in sentences if len(i) < max_len]

        return sentences


class TreeSentence(Sentence):
    """
    Args:
        transform (Tree):
            A Tree object.
        tree (nltk.Tree):
            A nltk.Tree object.
    """
    def __init__(self, transform, tree):
        super().__init__(transform)

        # the values contain words, pos tags, raw trees, and spans
        # the tree is first left-binarized before factorized
        # spans are the factorization of tree traversed in pre-order
        self.values = [
            *zip(*tree.pos()), tree,
            Tree.factorize(Tree.binarize(tree)[0])
        ]

    def __repr__(self):
        return self.values[-2].pformat(1000000)
