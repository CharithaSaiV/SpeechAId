import {
    Button,
    Dialog,
    DialogActions,
    DialogContent,
    DialogTitle,
    Typography,
  } from "@mui/material";
  import React from "react";
  
  export const ConfirmationDialog = ({
    open,
    dialogTitle,
    dialogMessage,
    handleCancel,
    handleConfirm,
    cancelButtonText,
    confirmButtonText,
  }) => {
    return (
      <Dialog
        fullWidth
        maxWidth={false}
        open={open}
        PaperProps={{
          sx: { maxWidth: 420, borderRadius: "15px" },
        }}
      >
      { dialogTitle &&  <DialogTitle>{dialogTitle }</DialogTitle>}
        <DialogContent>
          <Typography>
            {dialogMessage}
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button
            size="small"
            onClick={handleCancel}
            color="primary"
            variant="contained"
          >
            {cancelButtonText || "No"}
          </Button>
          <Button
            size="small"
            onClick={handleConfirm}
            color="error"
            variant="contained"
          >
            {confirmButtonText || "Yes"}
          </Button>
        </DialogActions>
      </Dialog>
    );
  };
  