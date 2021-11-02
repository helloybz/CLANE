import { Grid, Slider, useTheme } from "@mui/material";
import './Body.css';
import { Plot } from "../../component"


export function Body() {
    const theme = useTheme()
    return (
        <Grid
            container
            sx={{
                bgcolor: theme.palette.mode === 'dark' ? 'background.dark' : 'background.light',
            }}
        >
            <Grid item xs={8} component={Plot} />
            <Grid item xs={4} component={Slider} />
        </Grid>
    )
}