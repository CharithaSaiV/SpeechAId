import React from "react";
import {
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  Tooltip,
  Typography,
} from "@mui/material";
import { Icon } from "@iconify/react";

export const PhraseUploadDialog = ({
  open,
  dialogTitle,
  dialogMessage,
  handleClose,
  uploadContent,
}) => {
  return (
    <Dialog
      fullWidth
      maxWidth="sm"
      open={open}
      PaperProps={{
        sx: { maxWidth: 420, borderRadius: "15px" },
      }}
    >
      {dialogTitle && (
        <DialogTitle>
          {dialogTitle}
          <IconButton
            onClick={handleClose}
            style={{
              position: "absolute",
              right: 8,
              top: 8,
              // color: "#9e9e9e",
            }}
          >
            <Tooltip title="cancel">
            <Icon
              icon="zondicons:close-outline"
              sx={{ color: "red", fontWeight: 700 }}
            />
            </Tooltip>
          </IconButton>
        </DialogTitle>
      )}
      <DialogContent>
        <Typography>{dialogMessage}</Typography>
        <div style={{ marginTop: "20px" }}>{uploadContent}</div>
      </DialogContent>
    </Dialog>
  );
};
