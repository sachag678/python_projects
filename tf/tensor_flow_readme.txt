#tensor flow - readme (Notes on any intricacies)

------------------------------------------------------------- Variables
Always use tf.get_variable() instead of tf.Variable()
there is no initial value to get_variable()

W = tf.Variable(<initial-value>, name=<optional-name>)

W = tf.get_variable(name, shape=None, dtype=tf.float32, initializer=None,
       regularizer=None, trainable=True, collections=None)

Weird- Cannot assign i to a tensor because it is an integer but you can divide a tensor by it and you can add a numpy array to it

--------------------------------------------------------------------Arrays
cannot use plt.imshow(x) if x is a tensor, but result of plt.imshow(session.run(x)) will work so transpose takes an image and then transpoes it and then outputs it as an image

when you use a function from the tf object you must run it in session.run(func(x)) to get the output. 

------------------------------------------------------------------ Placeholders

here you dont have to create a model and intitialize variables

because the data is given at run time

-----------------------------------------------------------------
can you add something that is not a tensor to a tensor?

samples = tf.random_normal()
center = np.random.random(())

samples+=center? does this work

---------------------------------------------------------------- mnist
does the gradientdescenet minimize the cross entropy stop the loop at a certain point? like a convergence

--------------------------------------------------training - xavier
use tf.get_variable() 

and tf.contrib.layers.xavier_initializer()

------------------------------saving models
SAVE
w1 = tf.Variable(tf.truncated_normal(shape=[10]), name='w1')
w2 = tf.Variable(tf.truncated_normal(shape=[20]), name='w2')
tf.add_to_collection('vars', w1)
tf.add_to_collection('vars', w2)
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver.save(sess, 'my-model')
# `save` method will call `export_meta_graph` implicitly.
# you will get saved graph files:my-model.meta

RESTORE
sess = tf.Session()
new_saver = tf.train.import_meta_graph('my-model.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.get_collection('vars')
for v in all_vars:
    v_ = sess.run(v)
    print(v_)