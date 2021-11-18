import { Chart } from "react-google-charts";
import { data } from './zachary_deepwalk';
import { Grid, Slider, Typography, useTheme } from "@mui/material";
import { useEffect, useState } from "react";


export function Plot() {
    const theme = useTheme()
    const [iter, setIter] = useState(0)
    const [chartData, setChartData] = useState(data[iter])
    const onSliderChange = e => {
        setIter(e.target.value)
    }

    useEffect(() => (
        setChartData(data[iter])
    ), [iter])

    return (
        <Grid container item xs={12} justifyContent='center' >
            <Grid item xs={12} md={6}
                sx={{
                    mr: "1rem",
                    height: {
                        xs: '10rem',
                        lg: '30rem'
                    }
                }}>
                <Chart
                    width='100%'
                    height='100%'
                    chartType="ScatterChart"
                    loader={<div>Loading Chart</div>}
                    data={
                        [['x', 'y']].concat(chartData)
                    }

                    options={{
                        legend: 'none',
                        chartArea: {
                            "top": 0,
                            "right": 0,
                            "bottom": 0,
                            "left": 0
                        },
                        animation: {
                            duration: 1000,
                            easing: 'out',
                        },
                        backgroundColor: theme.palette.background.dark,
                        colors: [theme.palette.text.dark],
                        hAxis: {
                            "gridlines": {
                                "color": theme.palette.divider.dark,
                            },
                            "minorGridlines": {
                                "color": theme.palette.divider.dark,
                            },
                            "baseline": {
                                "color": theme.palette.divider.dark,
                            },
                            viewWindow: {
                                max: -8,
                                min: 8,
                            },
                        },
                        vAxis: {
                            "gridlines": {
                                "color": theme.palette.divider.dark,
                            },
                            "minorGridlines": {
                                "color": theme.palette.divider.dark,
                            },
                            "baseline": {
                                "color": theme.palette.divider.dark,
                            },
                            viewWindow: {
                                max: -8,
                                min: 8,
                            },
                        }
                    }}
                />
            </Grid>
            <Grid item xs={4}>
                <Typography sx={{ color: 'text.dark', fontSize: '2rem' }}>Iterations: {iter}</Typography>
                <Slider min={0} max={17} onChange={onSliderChange} marks />
            </Grid>
        </Grid >
    )
}