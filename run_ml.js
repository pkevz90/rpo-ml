import { log } from 'console';
import * as fs from 'fs';
import { dotDivide } from 'mathjs';
import * as tf from '@tensorflow/tfjs';
// import { ModelStoreManagerRegistry } from '@tensorflow/tfjs-core/dist/io/model_management.js';
let data
try {
    data = fs.readFileSync('./data.txt', 'utf8')
} catch (err) {
    console.error(err)
}
data = data.split('break')
data = data.filter(d => d !== '')
data = data.map(d => {
    let sData = d.split(',')
    return {x: Number(sData[0]), y: Number(sData[1]), tof: Number(sData[2]), dX: Number(sData[3]), dY: Number(sData[4])}
})
function convertToTensor(data) {
    return tf.tidy(() => {
        tf.util.shuffle(data)

        const inputs = data.map(d => [d.x, d.y, d.tof])
        const labels = data.map(d => [d.dX, d.dY])

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 3])
        const labelTensor = tf.tensor2d(labels, [labels.length, 2])
        
        const inputMax = inputTensor.max()
        const inputMin = inputTensor.min()
        const labelMax = labelTensor.max()
        const labelMin = labelTensor.min()

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))
        return {
            inputs: normalizedInputs,
            labels: normalizedLabels,
            // Return min/max bounds to use later
            inputMax,
            inputMin,
            labelMax,
            labelMin
        }
    })
}

function createModel() {
    const model = tf.sequential()

    model.add(tf.layers.dense({inputShape: [3], units: 50, activation: 'relu'}))
    model.add(tf.layers.dropout(0.5))
    model.add(tf.layers.dense({units: 200, activation: 'relu'}))
    model.add(tf.layers.dropout(0.5))
    model.add(tf.layers.dense({units: 100, activation: 'relu'}))
    model.add(tf.layers.dropout(0.5))
    model.add(tf.layers.dense({units: 50, activation: 'relu'}))
    model.add(tf.layers.dropout(0.5))
    model.add(tf.layers.dense({units: 2}))

    return model
  }
convertToTensor(data)
createModel()
const model = createModel()
async function run(data, model) {
    const tensorData = convertToTensor(data)
    const {inputs, labels} = tensorData
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse']
    })
    const batchSize = 100
    const epochs = 5
    await model.fit(inputs, labels, {
        batchSize,
        epochs,
        callbacks: {onEpochEnd}
    })
    console.log('Done Training');
    // await model.save('file://' + process.cwd() + '/newModel')
    await model.save('file://./model');
    const {inputMax, inputMin, labelMax, labelMin} = tensorData

    let x = tf.tensor([[10, 0, 1]])
    let normx = x.sub(inputMin).div(inputMax.sub(inputMin))
    let pred = model.predict(normx)
    let unnormpred = pred.mul(labelMax.sub(labelMin)).add(labelMin)
    console.log(x.arraySync(), unnormpred.arraySync());
    x = tf.tensor([[20, 20, 1]])
    normx = x.sub(inputMin).div(inputMax.sub(inputMin))
    pred = model.predict(normx)
    unnormpred = pred.mul(labelMax.sub(labelMin)).add(labelMin)
    console.log(x.arraySync(), unnormpred.arraySync());
    x = tf.tensor([[-20, 20, 1]])
    normx = x.sub(inputMin).div(inputMax.sub(inputMin))
    pred = model.predict(normx)
    unnormpred = pred.mul(labelMax.sub(labelMin)).add(labelMin)
    console.log(x.arraySync(), unnormpred.arraySync());
    x = tf.tensor([[-10,-15, 1]])
    normx = x.sub(inputMin).div(inputMax.sub(inputMin))
    pred = model.predict(normx)
    unnormpred = pred.mul(labelMax.sub(labelMin)).add(labelMin)
    console.log(x.arraySync(), unnormpred.arraySync());
}
function onEpochEnd(batch, logs) {
    console.log('Epoch',batch,'Accuracy', logs.loss)
}

run(data, model)