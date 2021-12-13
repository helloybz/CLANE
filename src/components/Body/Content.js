import { Grid, List, ListItem, Table, TableBody, TableCell, TableHead, TableRow, Typography } from "@mui/material";
import MathJax from 'react-mathjax2';

export function Content({ type, content, language }) {

    if (type === 'list') {
        var sentences = null;
        if (language === 'KOR') {
            sentences = content.kor.split('\n')
        } else {
            sentences = content.eng.split('\n')
        }
        return (
            <Grid container component={List}>
                {sentences.map((sentence, i) => (
                    <Grid key={i} item xs={12} component={ListItem}>
                        <Typography >
                            - {sentence}
                        </Typography>
                    </Grid>
                ))}
            </Grid>
        )
    } else if (type === 'paragraphs') {
        var paragraphs = null;
        if (language === 'KOR') {
            paragraphs = content.kor.split('\n')
        } else {
            paragraphs = content.eng.split('\n')
        }
        return (
            <Grid container component={List}>
                {paragraphs.map((paragraph, i) => {
                    var paragraph_trimmed = paragraph.trim()
                    if (paragraph_trimmed.startsWith("$$")) {
                        return (
                            <Grid item key={i} xs={12} component={ListItem}
                                justifyContent='center'
                                sx={{ overflow: 'auto' }}
                            >
                                <MathJax.Context
                                    input='tex'
                                    script="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.0.0/es5/latest?tex-mml-chtml.js"
                                    options={{
                                        tex: {
                                            packages: { '[+]': ['amsmath'] },
                                            tags: 'ams',
                                            tagIndent: "2em",
                                            tagSide: "right",
                                        }
                                    }}
                                >
                                    <MathJax.Text text={paragraph_trimmed} />
                                </MathJax.Context>
                            </Grid>
                        )

                    } else if (paragraph_trimmed.includes("$")) {

                        return (
                            <Grid item key={i} xs={12} component={ListItem}
                                sx={{ fontSize: '5rem' }}>
                                <MathJax.Context
                                    input='ascii'
                                    options={{
                                        asciimath2jax: {
                                            useMathMLspacing: true,
                                            delimiters: [["$", "$"]],
                                            preview: "none",
                                        }
                                    }}>
                                    <Typography paragraph>
                                        <MathJax.Text text={paragraph_trimmed} />
                                    </Typography>
                                </MathJax.Context>
                            </Grid>
                        )
                    }
                    else {
                        return (<Grid item key={i} xs={12} component={ListItem}>
                            <Typography paragraph>
                                {paragraph}
                            </Typography>
                        </Grid>)
                    }
                })}
            </Grid>
        )
    } else if (type === 'table') {
        return (
            <Grid item xs={12} sx={{ overflow: 'auto' }} >
                <Typography
                    variant="h5">{content.eng.title}</Typography>
                <Table>
                    <TableHead>
                        <TableRow>
                            {content.eng.header.map((name, i) => (
                                <TableCell key={i}>{name}</TableCell>
                            ))}
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {content.eng.rows.map((row, i) => (
                            <TableRow key={i}>
                                {row.map((val, j) => (
                                    <TableCell key={j}>{val}</TableCell>
                                ))}
                            </TableRow>
                        ))}
                    </TableBody>
                </Table>
            </Grid>
        )
    } else {
        return (
            <div>Error</div>
        )
    }
}


