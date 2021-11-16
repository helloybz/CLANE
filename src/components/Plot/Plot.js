import { Chart } from "react-google-charts";
import { data } from './zachary_deepwalk';
import { Grid, Slider } from "@mui/material";
import { useEffect, useState } from "react";


export function Plot() {
    const [iter, setIter] = useState(0)
    const [chartData, setChartData] = useState(data[iter])
    const onSliderChange = e => {
        setIter(e.target.value)
    }

    useEffect(() => (
        setChartData(data[iter])
    ), [iter])

    return (
        <Grid container>
            <Grid item xs={12}>
                <Chart
                    width={500}
                    height={500}
                    chartType="ScatterChart"
                    loader={<div>Loading Chart</div>}
                    data={
                        [['x', 'y']].concat(chartData)
                    }

                    options={{
                        legend: 'none',
                        title: "Applying CLANE to Deepwalk of Zachary Karate Club",
                        titleTextStyle: { color: 'rgb(243, 246, 249)' },
                        chartArea: {
                            "top": 50,
                            "right": 50,
                            "bottom": 50,
                            "left": 50
                        },
                        animation: {
                            duration: 1000,
                            easing: 'out',
                        },
                        backgroundColor: 'rgb(13, 25, 40)',
                        colors: ['rgb(243, 246, 249)'],
                        hAxis: {
                            "gridlines": {
                                "color": 'rgb(24, 47, 75)',
                            },
                            "minorGridlines": {
                                "color": 'rgb(24, 47, 75)',
                            },
                            "baseline": {
                                "color": 'rgb(24, 47, 75)',
                            },
                            viewWindow: {
                                max: -8,
                                min: 8,
                            },
                        },
                        vAxis: {
                            "gridlines": {
                                "color": 'rgb(24, 47, 75)',
                            },
                            "minorGridlines": {
                                "color": 'rgb(24, 47, 75)',
                            },
                            "baseline": {
                                "color": 'rgb(24, 47, 75)',
                            },
                            viewWindow: {
                                max: -8,
                                min: 8,
                            },
                        }
                    }}
                />
            </Grid>
            <Grid item xs={6}>
                <Slider min={0} max={17} onChange={onSliderChange} />
            </Grid>
        </Grid >
    )
}