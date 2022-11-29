<template>
  <div>
    <h1 class="visual">Визуализация данных</h1>
    <Loader v-if="loading">
    </Loader>
    <div class="flex">
      <Line :chart-options="chartOptions" :chart-data="chartData" class="chart" />
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

const chartData = ref(null)

chartData.value = {}

const loading = ref(false)
onMounted(async () => {
  loading.value = true
  const result = (await axios.get(props.url)).data
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
})

</script>
<style scoped>
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