import { Grid, useTheme } from "@mui/material";
import { Plot } from "../../components"


export function Body() {
    const theme = useTheme()
    return (
        <Grid
            container
            sx={{
                bgcolor: theme.palette.mode === 'dark' ? 'background.dark' : 'background.light',
            }}
        >
            <Grid item xs={8} >
                <Plot />
            </Grid>
        </Grid>
    )
}