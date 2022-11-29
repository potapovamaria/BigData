<template>
    <div>
        <Loader v-if="loading">
        </Loader>

        <div class="flex col">
            <Line :chart-options="chartOptions1" :chart-data="chartData1" class="chart" />
            <Line :chart-options="chartOptions2" :chart-data="chartData2" class="chart" />
            <Line :chart-options="chartOptions3" :chart-data="chartData3" class="chart" />
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

const chartOptions1 = ref({
    responsive: true,
    maintainAspectRatio: false
})

const chartOptions2 = ref({
    responsive: true,
    maintainAspectRatio: false
})

const chartOptions3 = ref({
    responsive: true,
    maintainAspectRatio: false
})

const chartData1 = ref(null)
const chartData2 = ref(null)
const chartData3 = ref(null)
chartData1.value = {}
chartData2.value = {}
chartData3.value = {}

const loading = ref(false)
onMounted(async () => {
    loading.value = true
    const result = (await axios.get(props.url)).data
    loading.value = false
    console.log(result.trend)
    chartData1.value = {
        labels: Object.keys(result.trend),
        datasets: [{
            label: "Тренд данных",
            borderColor: 'skyblue',
            borderWidth: 3,
            pointRadius: 2,
            data: Object.values(result.trend)
        }]
    }
    chartData2.value = {
        labels: Object.keys(result.seasonal),
        datasets: [{
            label: "Сезонность данных",
            borderColor: 'skyblue',
            borderWidth: 3,
            pointRadius: 2,
            data: Object.values(result.seasonal)
        }]
    }
    chartData3.value = {
        labels: Object.keys(result.resid),
        datasets: [{
            label: "Случайная составляющая данных",
            borderColor: 'skyblue',
            borderWidth: 3,
            pointRadius: 2,
            data: Object.values(result.resid)
        }]
    }
})

</script>
<style scoped>
.flex {
    display: flex;
}

.col {
    flex-direction: column;
}

.chart {
    padding: 10px;
    flex: 1;
}
</style>