<template>
  <div>
    <div class="flex row">

      <div class="flex col">
        <button v-for="chart, i in charNames" @click="onClick(i)" :class="currentChart == i ? 'selected' : ''">
          {{ chart }}
        </button>
      </div>
      <div class="chart">
        <HelloWorld v-if="currentChart == 0" url="http://localhost:5000/getdata" />
        <WeekDay v-if="currentChart == 1" url="http://localhost:5000/getWeekday" />
        <Day v-if="currentChart == 2" url="http://localhost:5000/getDay" />
        <Decomposition v-if="currentChart == 3" url="http://localhost:5000/getDecomposition" />
        <Price v-if="currentChart == 4" url="http://localhost:5000/getPrice" />
        <Prediction v-if="currentChart == 5" url="http://localhost:5000/prediction/payment" />
      </div>
    </div>
  </div>

</template>

<script setup>
import { ref } from 'vue';
import Day from './components/Day.vue';
import HelloWorld from './components/HelloWorld.vue'
import WeekDay from './components/WeekDay.vue';
import Decomposition from './components/Decomposition.vue'
import Price from './components/Price.vue';
import Prediction from './components/Prediction.vue';
const charNames = ref(["Данные", "День недели", "Число месяца", "Декомпозиция", "Тарифы", "Прогнозирование"])
const currentChart = ref(0)

function onClick(i) {
  currentChart.value = i
}
</script>

<style scoped>
.flex {
  display: flex;
}

.selected {
  background-color: aquamarine;
}

.row {
  flex-direction: row;
}

.col {
  flex-direction: column;
}

.chart {
  flex: 1;
}
</style>