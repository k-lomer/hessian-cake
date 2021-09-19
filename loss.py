import tensorflow as tf


class NN_Loss:
    """
    Neural Network loss class, should be callable
    """
    def __init__(self, model, loss_func, inputs=None, targets=None):
        """
        :param model: a keras Model or similar class
        :param loss_func: the actual function
        :param inputs: tf.Tensor: input values
        :param targets: tf.Tensor: expected output values
        Note the order of inputs and targets should be the same
        """
        self.model = model
        self.inputs = inputs
        self.targets = targets
        self.loss_func = loss_func

        self.subsamples = None

    def __call__(self, inputs=None, targets=None):
        """
        Evaluate the loss function on given data
        :param inputs: tf.Tensor: input values, optional, else will use current stored values
        :param targets: tf.Tensor: expected output values, optional, else will use current stored values
        Note the order of inputs and targets should be the same
        """
        if inputs is None:
            inputs = self.inputs
        if targets is None:
            targets = self.targets

        logits = self.model(inputs)
        return tf.reduce_mean(input_tensor=self.loss_func(logits, targets))

    def subsampled_loss(self):
        """
        Call the loss function on subsampled set of stored values
        """
        inputs = self.inputs[:self.subsamples]
        targets = self.targets[:self.subsamples]

        logits = self.model(inputs)
        return tf.reduce_mean(input_tensor=self.loss_func(logits, targets))

    def update_data(self, inputs, targets):
        """
        Update the stored data
        :param inputs: tf.Tensor: input values
        :param targets: tf.Tensor: expected output values
        Note the order of inputs and targets should be the same
        """
        self.inputs = inputs
        self.targets = targets
