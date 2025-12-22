PowerGridSynth (Power grid synthesizer) is an open source project written in Python, dedicated to build **synthetic power grids**. The goal of the project is to provide realistic yet synthetic power grids for grid modeling, simulation and analysis, with the ultimate goal of building **Foundation Models** for power grids. It is part of [LF Energy](https://lfenergy.org), a Linux Foundation focused on energy sector. This project is supported by [**AI-EFFECT** (Artificial Intelligence Experimentation Facility For the Energy Sector)](https://ai-effect.eu/).

It contains two main parts:
- grid topology generation,
- grid data generation. 

# topology generator

```{eval-rst}
.. autoclass:: powergrid_synth.InputConfigurator
   :members:
   :undoc-members:
   :show-inheritance:
```


```{eval-rst}
.. autoclass:: powergrid_synth.PowerGridGenerator
   :members:
   :undoc-members:
   :show-inheritance:
```



# grid data generator

```{eval-rst}
.. autoclass:: powergrid_synth.BusTypeAllocator
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.CapacityAllocator
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.LoadAllocator
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.TransmissionLineAllocator
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.GenerationDispatcher
   :members:
   :undoc-members:
   :show-inheritance:
```

# Utilities
```{eval-rst}
.. autoclass:: powergrid_synth.GridVisualizer
   :members:
   :undoc-members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: powergrid_synth.GridExporter
   :members:
   :undoc-members:
   :show-inheritance:
```
