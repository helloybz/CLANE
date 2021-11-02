import { Button, Grid, Typography } from "@mui/material";
import { GitHub } from "@mui/icons-material"
import './Header.css';
import { useTheme } from "@mui/system";


export function Header() {
    const theme = useTheme()
    return (
        <Grid
            container
            alignItems="baseline"
            sx={{
                bgcolor: theme.palette.mode === 'dark' ? 'background.dark' : 'background.light',
                padding: '1rem',
                borderBottom: 1,
                borderBottomColor: theme.palette.mode === 'dark' ? 'divider.dark' : 'divider.light'
            }}
        >
            <Grid item xs={3}>
                <Button
                    href="/"
                    className="brandButton"
                    sx={{
                        border: 1,
                        borderColor: theme.palette.mode === 'dark' ? 'divider.dark' : 'divider.light'
                    }}
                >
                    <Typography variant="h5">
                        CLANE
                    </Typography>
                </Button>
            </Grid>
            <Grid item xs></Grid>
            <Grid item xs={2}>
                <Button
                    href="https://github.com/helloybz/CLANE"
                    className="githubButton"
                    size="large"
                    startIcon={<GitHub />}
                    sx={{
                        border: 1,
                        borderColor: theme.palette.mode === 'dark' ? 'divider.dark' : 'divider.light'
                    }}
                    component={Typography}
                >
                    Github
                </Button>
            </Grid>
        </Grid >
    )
}