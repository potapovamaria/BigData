<template>
    <div>
        <Loader v-if="loading">
        </Loader>
        <div class="flex">
            <Bar :chart-options="chartOptions" :chart-data="chartData" class="chart" />
        </div>
    </div>
</template>
  
<script setup>
import { onMounted, ref } from 'vue';
import axios from 'axios';
import Loader from "./Loader.vue"
import { Bar } from 'vue-chartjs'
import { Chart as ChartJS, Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale } from 'chart.js'

ChartJS.register(Title, Tooltip, Legend, BarElement, CategoryScale, LinearScale)

const props = defineProps(["url"])

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

        labels: ['Понедельник', 'Вторник', 'Среда', 'Четверг', 'Пятница', 'Суббота', 'Воскресенье'],
        datasets: [{
            label: 'Количество максимальных платежей по дням недели',
            borderColor: 'skyblue',
            borderWidth: 3,
            data: Object.values(result),
            backgroundColor: 'skyblue'
        }]
    }
})

</script>
<style scoped>
.flex {
    display: flex;
}

.chart {
    flex: 1;
}
</style>