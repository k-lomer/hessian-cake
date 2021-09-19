import tensorflow as tf
from tensorflow.keras.metrics import categorical_accuracy

def train(model, data, optimizer, loss, log, epochs, batch_size, batch_loss=True, accuracy=False):
    (x_train, y_train), (x_test, y_test) = data
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    test_loss = split_test_loss(loss, x_test, y_test)
    if accuracy:
        test_acc = categorical_accuracy(y_test, model(x_test)).numpy().mean()
        log.log_value("test_acc", float(test_acc))

    log.log_value("test_loss", float(test_loss))
    print("loss",test_loss)
    try:
        log.log_value("CG Iter", optimizer.cg_iters)
    except AttributeError: # for first order methods
        pass
    log.write_log(batch=True, epoch=True)
    i=0

    for epoch in range(epochs):
        batched_data = train_data.shuffle(buffer_size=batch_size*4).batch(batch_size)
        for batch_x, batch_y in batched_data:
            loss.update_data(batch_x, batch_y)

            i += 1
            print(i)

            log.start_clock()
            optimizer.minimize()
            log.stop_clock()

            log.log_value("sweeps", optimizer.sweeps)
            if batch_loss:
                try:
                    log.log_value("CG Iter", optimizer.cg_iters)
                except AttributeError:  # for first order methods
                    pass
                test_loss = split_test_loss(loss, x_test, y_test)
                print("test loss: ", test_loss)
                log.log_value("test_loss", float(test_loss))
                log.write_log(batch=True)
                if accuracy:
                    test_acc = categorical_accuracy(y_test, model(x_test)).numpy().mean()
                    log.log_value("test_acc", float(test_acc))
                    print("test accuracy", float(test_acc))

        if not batch_loss:
            test_loss = loss(x_test, y_test)
            log.log_value("test_loss", float(test_loss))

        log.write_log(epoch=True)


def split_test_loss(loss, x_test, y_test, parts=10):
    assert(len(x_test) % parts == 0)
    chunk = len(x_test) // parts
    accumulative_loss = 0
    for i in range(parts):
        x = x_test[chunk * i: chunk * (i + 1)]
        y = y_test[chunk * i: chunk * (i + 1)]
        accumulative_loss += loss(x, y)
    return accumulative_loss / parts

def split_test_acc(model, x_test, y_test, parts=10):
    assert(len(x_test) % parts == 0)
    chunk = len(x_test) // parts
    accumulative_acc = 0
    for i in range(parts):
        x = x_test[chunk * i: chunk * (i + 1)]
        y = y_test[chunk * i: chunk * (i + 1)]
        accumulative_acc += categorical_accuracy(y_test, model(x_test)).numpy().mean()
    return accumulative_acc / parts











