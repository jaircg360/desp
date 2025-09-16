import React, { useState } from 'react';
import Dashboard from './components/Dashboard';

export default function App(){
  const [modelsUpdated, setModelsUpdated] = useState(false);
  return (
    <div className="container-app">
      <Dashboard onModelsUpdate={()=>setModelsUpdated(m=>!m)} />
    </div>
  );
}
