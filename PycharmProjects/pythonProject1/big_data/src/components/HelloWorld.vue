<template>
  <div class="popup" v-if="isShow">
    <header class="modal-header">
      <slot name="header">
        Enter dates for visualisation
      </slot>
    </header>
    <section class="modal-body">
      <slot name="body">
        <label class="label-from">from</label>
        <input type="date" v-model="date1" id="start" name="trip-start" min="2012-01-01" max="2022-03-30">
        <label class="label-to">to</label>
        <input type="date" v-model="date2" id="end" name="trip-end" min="2012-01-02" max="2022-03-31">
      </slot>
    </section>
    <button @click="hide">Submit</button>
  </div>
  <div>
    <h1 class="visual">Визуализация данных</h1>
    <Loader v-if="loading">
    </Loader>
    <div class="flex">
      <Line :chart-options="chartOptions" :chart-data="chartData" class="chart" />
    </div>
    <div class="flex">
      <button @click="show" v-if="isShowButon">New visual</button>
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref } from 'vue';
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
const isShowButon = ref(false)
const isShow = ref(true)
const chartData = ref(null)
function show() {
  isShow.value = true
  isShowButon.value = false
}
chartData.value = {}
const date1 = ref("2012-01-01")
const date2 = ref("2022-03-31")
const loading = ref(false)

async function hide() {
  isShowButon.value = true
  isShow.value = false
  loading.value = true
  const dataPost = {
    start_date: new Date(date1.value).toLocaleDateString("RU"),
    end_date: new Date(date2.value).toLocaleDateString("RU"),
  };
  const result = (await axios.post(props.url, dataPost)).data
  loading.value = false
  console.log(result)
  chartData.value = {
    labels: Object.keys(result.PAY),
    datasets: [{
      label: "График платежей",
      borderColor: 'skyblue',
      borderWidth: 3,
      pointRadius: 0.5,
      data: Object.values(result.PAY)
    }]
  }
}

// onMounted(async () => {
//   loading.value = true
//   const result = (await axios.get(props.url)).data
//   loading.value = false
//   console.log(result)
//   chartData.value = {
//     labels: Object.keys(result.PAY),
//     datasets: [{
//       label: "График платежей",
//       borderColor: 'skyblue',
//       borderWidth: 3,
//       pointRadius: 0.5,
//       data: Object.values(result.PAY)
//     }]
//   }
// })

</script>
<style scoped>
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

.flex {
  display: flex;
}

.visual {
  text-align: center;
}

.chart {
  flex: 1;
}
</style>