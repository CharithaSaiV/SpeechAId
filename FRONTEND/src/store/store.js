import { configureStore } from '@reduxjs/toolkit';
import dataReducer from './TableSlice'; // Adjust the path accordingly

const store = configureStore({
  reducer: {
    data: dataReducer,
  },
});

export default store;
