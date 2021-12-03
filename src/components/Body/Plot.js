import { useEffect, useState } from 'react';
import { Grid, Typography, Slider } from "@mui/material";
import Chart from 'react-google-charts';
import { useTheme } from '@mui/system';

export function Plot({ data, title }) {
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
        <Grid container item xs={12}>
            <Grid item xs={12}>
                <Chart
                    width={'100%'}
                    height={'100%'}
                    chartType="ScatterChart"
                    data={chartData}
                    options={{
                        title: title,
                        titleTextStyle: {
                            color: "white",
                        },
                        legend: {
                            position: "right",
                            textStyle: { color: 'white', fontSize: "2rem" },
                        },
                        backgroundColor: theme.palette.background.dark,
                        colors: ["magenta", "yellow", "grey", "cyan"],
                        hAxis: {
                            "gridlines": {
                                "color": theme.palette.border.dark,
                            },
                            "minorGridlines": {
                                "color": theme.palette.border.dark,
                            },
                            "baseline": {
                                "color": theme.palette.border.dark,
                            },
                            viewWindow: {
                                max: 50,
                                min: -50,
                            },
                        },
                        vAxis: {
                            "gridlines": {
                                "color": theme.palette.border.dark,
                            },
                            "minorGridlines": {
                                "color": theme.palette.border.dark,
                            },
                            "baseline": {
                                "color": theme.palette.border.dark,
                            },
                            viewWindow: {
                                max: 50,
                                min: -50,
                            },
                        }
                    }}
                />
            </Grid>
            <Grid item xs={12}>
                <Typography sx={{ color: 'text.dark', fontSize: '2rem' }}>Iterations: {iter}</Typography>
                <Slider min={0} max={61} onChange={onSliderChange} marks />
            </Grid>
        </Grid >
    )
}