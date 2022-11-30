<template>
    <div>
        <div class="popup" v-if="isShow">
            <header class="modal-header">
                <slot name="header">
                    Enter dates for prediction
                </slot>
            </header>
            <section class="modal-body">
                <slot name="body">
                    <label class="label-from">from</label>
                    <input type="date" v-model="date1" id="start" name="trip-start" min="2010-01-01" max="2022-12-31">
                    <label class="label-to">to</label>
                    <input type="date" v-model="date2" id="end" name="trip-end" min="2010-01-01" max="2022-12-31">
                </slot>
            </section>
            <footer class="modal-footer">
                <slot name="footer">
                    <input type="checkbox" id="checkbox" v-model="checked" />
                    <label for="checkbox">Add picks forecasting</label>
                </slot>
            </footer>
            <button @click="hide">Submit</button>
        </div>
        <div>
            <Loader v-if="loading">
            </Loader>
            <div class="flex">
                <Line :chart-options="chartOptions" :chart-data="chartData" class="chart" />
            </div>
        </div>
        <div class="flex">
            <button @click="show" v-if="isShowButon">New forecasting</button>
        </div>

    </div>

</template>

<script setup>
import { ref } from 'vue';
import axios from 'axios';
import Loader from "./Loader.vue"
import { Line } from 'vue-chartjs'
import {
    Chart as ChartJS,
    Title,
    Tooltip,
    Legend,
    LineElement,
    LinearScale,
    PointElement,
    CategoryScale
} from 'chart.js'

const props = defineProps(["url"])

ChartJS.register(
    Title,
    Tooltip,
    Legend,
    LineElement,
    LinearScale,
    PointElement,
    CategoryScale
)

const chartOptions = ref({
    responsive: true,
    maintainAspectRatio: false
})

const chartData = ref(null)

chartData.value = {}

const loading = ref(false)
const isShow = ref(true)
const checked = ref(false)
const isShowButon = ref(false)
const date1 = ref("2021-11-01")
const date2 = ref("2022-11-01")
function show() {
    isShow.value = true
    isShowButon.value = false
}
async function hide() {
    isShow.value = false

    loading.value = true
    let checkPick = "1"
    if (checked.value == true) {
        checkPick = "2"
    }
    const dataPost = {
        model_num: "1",
        start_date: new Date(date1.value).toLocaleDateString("RU"),
        end_date: new Date(date2.value).toLocaleDateString("RU"),
        enter_pick: checkPick
    };
    const result = (await axios.post(props.url, dataPost)).data
    loading.value = false
    console.log(result)

    const dataOrig = {
        start_date: date1.value,
        end_date: date2.value
    }

    const resultOrig = (await axios.post("http://localhost:5000/getdatapred", dataOrig)).data
    loading.value = false
    console.log(resultOrig)

    if (resultOrig == "No data") {
        chartData.value = {
            labels: Object.keys(result.PAY),
            datasets: [{
                label: 'Prediction data',
                pointRadius: 1,
                borderColor: '#77b7cd', // цвет линии

                data: Object.values(result.PAY)
            }]
        }
    } else {
        chartData.value = {
            labels: Object.keys(result.PAY),
            datasets: [{
                label: 'Prediction data',
                pointRadius: 1,
                borderColor: '#77b7cd', // цвет линии

                data: Object.values(result.PAY)
            },
            {
                label: 'Real data',
                pointRadius: 1,
                borderColor: '#7777cd', // цвет линии

                data: Object.values(resultOrig.PAY)
            },
            ]
        }
    }


    isShowButon.value = true
}

</script>

<style>
.popup {
    width: 400px;
    height: 400px;
    position: fixed;
    top: 100px;
    bottom: 0;
    left: 450px;
    right: 0;
    background-color: rgba(0, 0, 0, 0.3);
    background: #FFFFFF;
    box-shadow: 2px 2px 20px 1px;
    overflow-x: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}

.buttonNew {
    position: fixed;
    top: 300px;
    bottom: 0;
    left: 450px;
    right: 0;
}

.label-to,
.label-from {
    justify-content: center;
    padding: 15px;
    display: flex;
}

.modal-header,
.modal-footer {
    padding: 15px;
    display: flex;
}

.modal-header {
    border-bottom: 1px solid #eeeeee;
    color: #4AAE9B;
    justify-content: space-between;
}

.modal-footer {
    border-top: 1px solid #eeeeee;
    justify-content: flex-end;
}

.date-start {
    padding: 10px;
}

.date-end {
    padding: 10px;
}

.chart {
    flex: 1;

}

.modal-body {
    position: relative;
    padding: 20px 10px;
}
</style>